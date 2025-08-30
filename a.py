import os
import sys
import numpy as np
import nibabel as nib
import pydicom
from skimage import measure
from skimage.measure import label
from skimage.transform import resize
import trimesh
import onnxruntime as ort

# =========================
# CARGA DE VOLÚMENES
# =========================

def load_nifti(path):
    """
    Carga un NIfTI y devuelve:
      volume: ndarray (Z, Y, X) en float32
      affine: matriz 4x4 de nib (vox -> mundo) en el orden (x,y,z)
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    # nib da data en (X, Y, Z). Convertimos a (Z, Y, X) para usar marching_cubes coherente.
    volume = np.transpose(data, (2, 1, 0))  # (Z, Y, X)
    return volume, affine

def load_dicom_series(dicom_dir):
    """
    Carga una serie DICOM desde un directorio (archivos .dcm) y devuelve:
      volume: ndarray (Z, Y, X) float32
      affine: 4x4 aproximada a partir de PixelSpacing, SliceThickness, etc.
    """
    import glob
    files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if not files:
        raise ValueError("No se encontraron archivos .dcm en " + dicom_dir)

    # Leer y ordenar por InstanceNumber (si existe)
    slices = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=False, force=True)
        slices.append(ds)
    try:
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    except Exception:
        pass

    # Construir volumen (Y, X, Z) y luego convertir a (Z, Y, X)
    rows = int(slices[0].Rows)
    cols = int(slices[0].Columns)
    Nz = len(slices)
    vol = np.zeros((rows, cols, Nz), dtype=np.float32)
    for i, s in enumerate(slices):
        vol[:, :, i] = s.pixel_array.astype(np.float32)

    # Espaciados
    ps = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    try:
        st = float(getattr(slices[0], "SliceThickness", 1.0))
    except Exception:
        st = 1.0
    # Affine aproximada (x->col, y->row, z->slice)
    affine = np.array([
        [ps[1], 0,      0,      0],
        [0,     ps[0],  0,      0],
        [0,     0,      st,     0],
        [0,     0,      0,      1],
    ], dtype=np.float32)

    volume = np.transpose(vol, (2, 0, 1))  # (Z, Y, X)
    return volume, affine

# =========================
# UTILIDADES DE GEOMETRÍA
# =========================

def apply_affine_transform(verts_xyz, affine):
    """
    Aplica matriz affine (vox->mundo) a una nube de puntos en orden (x,y,z).
    verts_xyz: (N,3) con columnas x,y,z
    """
    ones = np.ones((verts_xyz.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([verts_xyz.astype(np.float32), ones], axis=1)  # (N,4)
    out = hom @ affine.T
    return out[:, :3]

def combine_and_save_obj(meshes, output_obj_path):
    """
    Combina una lista de (verts, faces) en un único OBJ y lo guarda.
    verts en (N,3) float, faces en (M,3) int.
    """
    tri_list = []
    for (v, f) in meshes:
        tri_list.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
    if not tri_list:
        raise ValueError("No hay mallas para exportar")
    combined = trimesh.util.concatenate(tri_list)
    combined.export(output_obj_path)

# =========================
# SEGMENTACIÓN POR UMBRAL (BRAIN)
# =========================

def segment_objects(volume, threshold=1.0):
    """
    volume (Z,Y,X) -> binario por umbral, luego componentes conectados.
    Devuelve (labeled, num_labels).
    """
    mask = (volume >= float(threshold)).astype(np.uint8)
    labeled = label(mask, connectivity=1)
    num_labels = labeled.max()
    return labeled, num_labels

def labeled_to_meshes(labeled, num_labels, affine):
    """
    Para cada etiqueta 1..num_labels, ejecuta marching_cubes y aplica affine.
    Devuelve lista de (verts_world, faces).
    """
    meshes = []
    for i in range(1, num_labels + 1):
        m = (labeled == i).astype(np.float32)
        if m.sum() == 0:
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(m, level=0.5)
        except Exception:
            continue
        # verts está en (z,y,x). Reordenamos a (x,y,z) para affine:
        verts_xyz = verts[:, [2, 1, 0]]
        verts_world = apply_affine_transform(verts_xyz, affine)
        meshes.append((verts_world, faces.astype(np.int64)))
    return meshes

# =========================
# CONVERSIÓN PRINCIPAL (BRAIN)
# =========================

def convert_medical_to_obj(input_path, output_obj_path, is_dicom=False, threshold=1.0):
    """
    Caso 'brain': umbral + componentes + marching_cubes de cada objeto.
    """
    if is_dicom:
        volume, affine = load_dicom_series(input_path)
    else:
        volume, affine = load_nifti(input_path)

    labeled, num_objects = segment_objects(volume, threshold)
    meshes = labeled_to_meshes(labeled, num_objects, affine)
    combine_and_save_obj(meshes, output_obj_path)

# =========================
# CONVERSIÓN TUMOR (ONNX)
# =========================

def convert_tumor_to_obj(input_path, output_obj_path, is_dicom=False, onnx_model_path="unet3d.onnx"):
    """
    Caso 'tumor': usa modelo ONNX (UNet3D) que espera (1,4,128,128,128) NCDHW.
    - Reescala el volumen a 128^3 (sin normalización extra porque el ONNX
      ya incluye 'ImageInputLayer_Mean' en sus nodos iniciales).
    - Replica a 4 canales.
    - Inferencia -> máscara 128^3.
    - Marching cubes -> rescale a rejilla original -> affine -> OBJ.
    """
    # 1) Carga volumen (Z,Y,X) y affine
    if is_dicom:
        volume, affine = load_dicom_series(input_path)
    else:
        volume, affine = load_nifti(input_path)

    Z, Y, X = volume.shape

    # 2) Reescala a 128^3
    v128 = resize(volume.astype(np.float32), (128, 128, 128),
                  order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)

    # 3) NCDHW (1, 4, 128, 128, 128) replicando el mismo contraste
    x = np.stack([v128, v128, v128, v128], axis=0)[None, ...]

    # 4) Inferencia ONNX
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_model_path, providers=providers)
    inp_name = sess.get_inputs()[0].name
    y = sess.run(None, {inp_name: x})[0]

    # 5) Salida -> máscara binaria 128^3
    if y.ndim == 5 and y.shape[1] > 1:
        # multi-clase: 0=bg, 1=tumor
        mask128 = (np.argmax(y, axis=1)[0] == 1).astype(np.uint8)
    elif y.ndim == 5:
        mask128 = (y[0, 0] > 0.5).astype(np.uint8)
    else:
        mask128 = (y[0] > 0.5).astype(np.uint8)

    if mask128.sum() == 0:
        # evita error de marching_cubes
        combine_and_save_obj([], output_obj_path)  # lanzará ValueError
        return

    # 6) Marching cubes en 128^3
    verts, faces, _, _ = measure.marching_cubes(mask128.astype(np.float32), level=0.5)

    # 7) Escalar de vuelta: 128 -> (Z,Y,X)
    scale_zyx = np.array([Z / 128.0, Y / 128.0, X / 128.0], dtype=np.float32)
    verts_scaled = verts * scale_zyx  # (z,y,x) en rejilla original

    # 8) Reordenar a (x,y,z) y aplicar affine
    verts_xyz = verts_scaled[:, [2, 1, 0]]
    verts_world = apply_affine_transform(verts_xyz, affine)

    # 9) Guardar OBJ
    combine_and_save_obj([(verts_world, faces.astype(np.int64))], output_obj_path)

# =========================
# CLI
# =========================

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso:")
        print("  python a.py <input_path> <output.obj> <tipo> [segment] [threshold]")
        print("  tipo: dicom|nifti")
        print("  segment: brain|tumor  (por defecto brain)")
        print("  threshold: para brain (default 1.0)")
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
        onnx_path = os.getenv("TUMOR_ONNX_PATH", "unet3d.onnx")
        convert_tumor_to_obj(input_path, output_obj, is_dicom, onnx_model_path=onnx_path)
