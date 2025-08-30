# api.py
import os
import shutil
import uuid
import zipfile
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from a import convert_medical_to_obj, convert_tumor_to_obj

app = FastAPI(title="Medical → OBJ Converter")

def safe_name_root(filename: str) -> str:
    # quita .nii o .nii.gz, o extensión genérica
    name = filename or "volume"
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    base, _ = os.path.splitext(name)
    return base

def cleanup_paths(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

@app.post("/convert", summary="Convierte NIfTI (.nii/.nii.gz) o un ZIP DICOM a OBJ")
async def convert_endpoint(
    file: UploadFile = File(...),
    tipo: str = Form("nifti"),                # "nifti" o "dicom"
    segment: str = Form("brain")              # "brain" o "tumor"
):
    tipo = (tipo or "nifti").lower()
    if tipo not in ("nifti", "dicom"):
        raise HTTPException(400, "tipo debe ser 'nifti' o 'dicom'")

    segment = (segment or "brain").lower()
    if segment not in ("brain", "tumor"):
        raise HTTPException(400, "segment debe ser 'brain' o 'tumor'")

    # Directorio temporal de trabajo
    work_dir = tempfile.mkdtemp(prefix="conv_")
    name_root = safe_name_root(file.filename or "volume")

    # Guardar entrada
    if tipo == "nifti":
        # Acepta .nii o .nii.gz
        ext = ".nii.gz" if (file.filename or "").lower().endswith(".nii.gz") \
              else os.path.splitext(file.filename or "")[1] or ".nii"
        in_path = os.path.join(work_dir, f"{name_root}{ext}")
        with open(in_path, "wb") as f:
            f.write(await file.read())
    else:
        # DICOM: se espera un ZIP con .dcm adentro
        zip_path = os.path.join(work_dir, f"{name_root}.zip")
        with open(zip_path, "wb") as f:
            f.write(await file.read())
        dicom_dir = os.path.join(work_dir, "dicom")
        os.makedirs(dicom_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dicom_dir)
        except zipfile.BadZipFile:
            cleanup_paths(work_dir)
            raise HTTPException(400, "Para tipo=dicom envía un .zip con los .dcm dentro")
        in_path = dicom_dir

    out_obj = os.path.join(work_dir, f"{name_root}.obj")

    try:
        if segment == "brain":
            # flujo existente: umbral=1.0
            convert_medical_to_obj(in_path, out_obj, is_dicom=(tipo=="dicom"), threshold=1.0)
        else:
            # tumor por ONNX
            onnx_path = os.getenv("TUMOR_ONNX_PATH", "unet3d.onnx")
            convert_tumor_to_obj(in_path, out_obj, is_dicom=(tipo=="dicom"), onnx_model_path=onnx_path)
    except Exception as e:
        cleanup_paths(work_dir)
        raise HTTPException(500, f"Error en conversión: {e}")

    if not os.path.isfile(out_obj):
        cleanup_paths(work_dir)
        raise HTTPException(500, "No se generó el archivo OBJ")

    return FileResponse(
        path=out_obj,
        media_type="application/octet-stream",
        filename=f"{name_root}.obj",
        background=BackgroundTask(lambda: cleanup_paths(work_dir))
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")
