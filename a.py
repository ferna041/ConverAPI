import os
# ↓↓↓ Limita threads ANTES de importar numpy/onnxruntime (reduce RAM en instancias chicas)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import gc
import numpy as np
import nibabel as nib
import pydicom
from skimage import measure
from skimage.measure import label
from skimage.transform import resize
import trimesh
import onnxruntime as ort

# ===== ONNX session cache (evita recargar el grafo/pesos cada request) =====
_SESSION_CACHE = {}

def get_onnx_session(model_path: str | None):
    if not model_path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "unet3d.onnx")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo ONNX no encontrado: {model_path}")

    if model_path in _SESSION_CACHE:
        return _SESSION_CACHE[model_path]

    so = ort.SessionOptions()
    # Menos hilos → menos RSS y contención
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    # Desactivar arenas ayuda en contenedores con poca RAM
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    available = ort.get_available_providers()
    preferred = ("CUDAExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider")
    use_prov = [p for p in preferred if p in available]

    sess = ort.InferenceSession(model_path, sess_options=so, providers=use_prov or None)
    _SESSION_CACHE[model_path] = sess
    return sess

# =========================
# CARGA DE VOLÚMENES
# =========================

def load_nifti(path):
    """
    Carga NIfTI y devuelve:
      volume: ndarray (Z, Y, X) float32
      affine: 4x4 (vox->mundo, orden x,y,z)
    """
    img = nib.load(path)
    # float32 directo (evita buffer float64)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    # nib: (X,Y,Z) → a (Z,Y,X) para marching_cubes
    volume = np.transpose(data, (2, 1, 0)).astype(np.float32, copy=False)
    # libera referencia a data
    del data
    gc.collect()
    return volume, affine

def load_dicom_series(dicom_dir):
    """
    Carga serie DICOM:
      volume: (Z,Y,X) float32
      affine: 4x4 aproximada
    """
    import glob
    files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if not files:
        raise ValueError("No se encontraron .dcm en " + dicom_dir)

    slices = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=False, force=True)
        slices.append(ds)
    try:
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    except Exception:
        pass

    rows = int(slices[0].Rows); cols = int(slices[0].Columns); Nz = len(slices)
    vol = np.empty((rows, cols, Nz), dtype=np.float32)
    for i, s in enumerate(slices):
        # evita float64
        vol[:, :, i] = s.pixel_array.astype(np.float32, copy=False)

    ps = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    try:
        st = float(getattr(slices[0], "SliceThickness", 1.0))
    except Exception:
        st = 1.0
    affine = np.array([
        [ps[1], 0,      0, 0],
        [0,     ps[0],  0, 0],
        [0,     0,      st,0],
        [0,     0,      0, 1],
    ], dtype=np.float32)

    volume = np.transpose(vol, (2, 0, 1))  # (Z,Y,X)
    del vol, slices
    gc.collect()
    return volume, affine

# =========================
# UTILIDADES DE GEOMETRÍA
# =========================

def apply_affine_transform(verts_xyz, affine):
    """
    verts_xyz: (N,3) en (x,y,z) → aplica affine (vox->mundo)
    """
    ones = np.ones((verts_xyz.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([verts_xyz.astype(np.float32, copy=False), ones], axis=1)  # (N,4)
    out = hom @ affine.T
    return out[:, :3]

def combine_and_save_obj(meshes, output_obj_path):
    """
    meshes: lista de (verts, faces)
    Exporta un único OBJ. Usa process=False para no re-tesselar.
    """
    if not meshes:
        # crea un OBJ mínimo vacío válido
        with open(output_obj_path, "w") as f:
            f.write("# empty OBJ\n")
        return

    if len(meshes) == 1:
        v, f = meshes[0]
        tri = trimesh.Trimesh(vertices=v, faces=f.astype(np.int32, copy=False), process=False)
        tri.export(output_obj_path)
        return

    tri_list = [trimesh.Trimesh(vertices=v, faces=f.astype(np.int32, copy=False), process=False)
                for (v, f) in meshes]
    combined = trimesh.util.concatenate(tri_list)
    combined.export(output_obj_path)
    # libera
    del tri_list, combined
    gc.collect()

# =========================
# SEGMENTACIÓN POR UMBRAL (BRAIN)
# =========================

def segment_objects(volume, threshold=1.0):
    mask = (volume >= float(threshold)).astype(np.uint8, copy=False)
    labeled = label(mask, connectivity=1)
    num_labels = int(labeled.max())
    return labeled, num_labels

def labeled_to_meshes(labeled, num_labels, affine, min_voxels=0):
    """
    min_voxels>0 → ignora blobs minúsculos (reduce triángulos y RAM).
    """
    meshes = []
    for i in range(1, num_labels + 1):
        m = (labeled == i)
        vox = int(m.sum())
        if vox == 0 or (min_voxels > 0 and vox < min_voxels):
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(m.astype(np.float32, copy=False), level=0.5)
        except Exception:
            continue
        # (z,y,x) → (x,y,z)
        verts_xyz = verts[:, [2, 1, 0]]
        verts_world = apply_affine_transform(verts_xyz, affine)
        meshes.append((verts_world.astype(np.float32, copy=False),
                       faces.astype(np.int32, copy=False)))
    # libera intermedios
    del labeled
    gc.collect()
    return meshes

# =========================
# CONVERSIÓN PRINCIPAL (BRAIN)
# =========================

def convert_medical_to_obj(input_path, output_obj_path, is_dicom=False, threshold=1.0, min_voxels=0):
    """
    Caso 'brain': umbral + componentes + marching_cubes. min_voxels=0 para mantener comportamiento actual.
    """
    if is_dicom:
        volume, affine = load_dicom_series(input_path)
    else:
        volume, affine = load_nifti(input_path)

    labeled, num_objects = segment_objects(volume, threshold)
    meshes = labeled_to_meshes(labeled, num_objects, affine, min_voxels=min_voxels)
    combine_and_save_obj(meshes, output_obj_path)
    # libera
    del volume, affine, meshes
    gc.collect()

# =========================
# CONVERSIÓN TUMOR (ONNX)
# =========================

def convert_tumor_to_obj(input_path, output_obj_path, is_dicom=False, onnx_model_path=None):
    """
    Caso 'tumor': ONNX UNet3D NCDHW (1,4,128,128,128).
    Reescala a 128^3, replica canales, inferencia, marching_cubes,
    reescala a rejilla original, affine → OBJ.
    """
    # 1) Carga volumen
    if is_dicom:
        volume, affine = load_dicom_series(input_path)
    else:
        volume, affine = load_nifti(input_path)
    Z, Y, X = volume.shape

    # 2) Reescala a 128^3 (float32)
    v128 = resize(volume, (128, 128, 128), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32, copy=False)
    del volume; gc.collect()

    # 3) NCDHW (1,4,128,128,128) - replicando un solo contraste
    x = np.stack([v128, v128, v128, v128], axis=0)[None, ...]
    del v128; gc.collect()

    # 4) Sesión ONNX (cacheada)
    sess = get_onnx_session(onnx_model_path)
    inp_name = sess.get_inputs()[0].name

    # 5) Inferencia
    y = sess.run(None, {inp_name: x})[0]
    del x; gc.collect()

    # 6) Salida → máscara 128^3
    if y.ndim == 5 and y.shape[1] > 1:
        mask128 = (np.argmax(y, axis=1)[0] == 1).astype(np.uint8, copy=False)
    elif y.ndim == 5:
        mask128 = (y[0, 0] > 0.5).astype(np.uint8, copy=False)
    else:
        mask128 = (y[0] > 0.5).astype(np.uint8, copy=False)
    del y; gc.collect()

    if int(mask128.sum()) == 0:
        raise ValueError("La segmentación del tumor resultó vacía (0 vóxeles).")

    # 7) Marching cubes en 128^3
    verts, faces, _, _ = measure.marching_cubes(mask128.astype(np.float32, copy=False), level=0.5)
    del mask128; gc.collect()

    # 8) Reescala a rejilla original: (z,y,x) * [Z/128, Y/128, X/128]
    scale_zyx = np.array([Z/128.0, Y/128.0, X/128.0], dtype=np.float32)
    verts_scaled = (verts * scale_zyx).astype(np.float32, copy=False)
    del verts; gc.collect()

    # 9) (z,y,x) → (x,y,z) y affine
    verts_xyz = verts_scaled[:, [2, 1, 0]]
    del verts_scaled; gc.collect()

    verts_world = apply_affine_transform(verts_xyz, affine)

    # 10) Guardar OBJ
    combine_and_save_obj([(verts_world, faces.astype(np.int32, copy=False))], output_obj_path)
    del verts_world, faces
    gc.collect()

# =========================
# CLI para debug local
# =========================

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso:")
        print("  python a.py <input_path> <output.obj> <tipo> [segment] [threshold]")
        print("  tipo: dicom|nifti")
        print("  segment: brain|tumor (default brain)")
        print("  threshold: brain (default 1.0)")
        sys.exit(1)

    input_path = sys.argv[1]
    output_obj = sys.argv[2]
    tipo = sys.argv[3].lower()
    is_dicom = (tipo == "dicom")
    segment = sys.argv[4].lower() if len(sys.argv) > 4 else "brain"

    if segment == "brain":
        threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        convert_medical_to_obj(input_path, output_obj, is_dicom, threshold)
    else:
        onnx_path = os.getenv("TUMOR_ONNX_PATH")
        convert_tumor_to_obj(input_path, output_obj, is_dicom, onnx_model_path=onnx_path)
