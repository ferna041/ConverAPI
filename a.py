import os
import sys
import numpy as np
import nibabel as nib
import pydicom
from skimage import measure
from skimage.measure import label
import trimesh

# --- CARGA DE VOLUMENES ---

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    volume = np.transpose(data, (2, 1, 0))  # Z, Y, X
    return volume, affine

def load_dicom_series(folder_path):
    slices = []
    for f in os.listdir(folder_path):
        if f.endswith(".dcm"):
            slices.append(pydicom.dcmread(os.path.join(folder_path, f)))
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices])
    spacing = slices[0].PixelSpacing  # [row, col]
    thickness = float(slices[0].SliceThickness)

    # Crear affine aproximado (traslación + spacing)
    origin = slices[0].ImagePositionPatient  # (x, y, z) del primer slice
    affine = np.array([
        [0, 0, thickness, origin[2]],
        [0, spacing[1], 0, origin[1]],
        [spacing[0], 0, 0, origin[0]],
        [0, 0, 0, 1]
    ])
    return volume, affine

# --- SEGMENTACIÓN DE OBJETOS ---

def segment_objects(volume, threshold=1):
    mask = volume >= threshold
    labeled = label(mask, connectivity=1)
    num_objects = labeled.max()
    print(f"Detectados {num_objects} objeto(s) en el volumen.")
    return labeled, num_objects

# --- APLICAR TRANSFORMACIÓN ESPACIAL ---

def apply_affine_transform(verts, affine):
    verts_h = np.c_[verts, np.ones(len(verts))]  # Nx4
    verts_world = verts_h @ affine.T
    return verts_world[:, :3]

# --- MARCHING CUBES PARA CADA OBJETO ---

def labeled_to_meshes(labeled, num_objects, affine):
    meshes = []
    for obj_id in range(1, num_objects + 1):
        mask = (labeled == obj_id)
        if np.count_nonzero(mask) == 0:
            continue
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
        verts = apply_affine_transform(verts, affine)
        meshes.append((verts, faces))
        print(f"  Objeto {obj_id}: {len(verts)} vértices, {len(faces)} caras")
    return meshes

# --- COMBINAR Y GUARDAR OBJ ---

def combine_and_save_obj(meshes, filename):
    combined = trimesh.util.concatenate([
        trimesh.Trimesh(vertices=verts, faces=faces)
        for verts, faces in meshes
    ])
    combined.remove_duplicate_faces()
    combined.remove_degenerate_faces()
    combined.fix_normals()
    combined.export(filename)
    print(f"Malla combinada exportada como: {filename}")

# --- FUNCIÓN PRINCIPAL DE CONVERSIÓN ---

def convert_medical_to_obj(input_path, output_obj_path, is_dicom=False, threshold=1):
    if is_dicom:
        volume, affine = load_dicom_series(input_path)
    else:
        volume, affine = load_nifti(input_path)

    labeled, num_objects = segment_objects(volume, threshold)
    meshes = labeled_to_meshes(labeled, num_objects, affine)
    combine_and_save_obj(meshes, output_obj_path)

# --- CLI MAIN ---

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python medical_to_obj.py <input_path> <output_path> <tipo> [threshold]")
        print("tipo: dicom o nifti")
        print("threshold: opcional (por defecto 1)")
        sys.exit(1)

    input_path = sys.argv[1]
    output_obj = sys.argv[2]
    tipo = sys.argv[3].lower()
    is_dicom = tipo == "dicom"
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    print(f"Convirtiendo: {input_path} -> {output_obj}")
    print(f"Tipo: {'DICOM' if is_dicom else 'NIfTI'}, threshold: {threshold}")
    convert_medical_to_obj(input_path, output_obj, is_dicom, threshold)
