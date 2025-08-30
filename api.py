import os
import shutil
import zipfile
import tempfile
import traceback
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from a import convert_medical_to_obj, convert_tumor_to_obj

app = FastAPI(title="Medical → OBJ Converter")

def safe_name_root(filename: str) -> str:
    if not filename:
        return "volume"
    name = filename
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    base, _ = os.path.splitext(name)
    return base or "volume"

def cleanup_paths(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

@app.post("/convert", summary="Convierte NIfTI (.nii/.nii.gz) o ZIP DICOM a OBJ (brain|tumor)")
async def convert_endpoint(
    file: UploadFile = File(...),
    tipo: str = Form("nifti"),           # "nifti" | "dicom"
    segment: str = Form("brain")         # "brain" | "tumor"
):
    tipo = (tipo or "nifti").lower()
    if tipo not in ("nifti", "dicom"):
        raise HTTPException(400, "tipo debe ser 'nifti' o 'dicom'")

    segment = (segment or "brain").lower()
    if segment not in ("brain", "tumor"):
        raise HTTPException(400, "segment debe ser 'brain' o 'tumor'")

    work_dir = tempfile.mkdtemp(prefix="conv_")
    name_root = safe_name_root(getattr(file, "filename", "") or "volume")

    try:
        # ======= guarda input por streaming (evita subir todo a RAM) =======
        if tipo == "nifti":
            ext = ".nii.gz" if (file.filename or "").lower().endswith(".nii.gz") \
                  else os.path.splitext(file.filename or "")[1] or ".nii"
            in_path = os.path.join(work_dir, f"{name_root}{ext}")
            with open(in_path, "wb") as f:
                await file.seek(0)
                shutil.copyfileobj(file.file, f, length=1024*1024)  # 1 MB chunks
        else:
            zip_path = os.path.join(work_dir, f"{name_root}.zip")
            with open(zip_path, "wb") as f:
                await file.seek(0)
                shutil.copyfileobj(file.file, f, length=1024*1024)
            dicom_dir = os.path.join(work_dir, "dicom")
            os.makedirs(dicom_dir, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(dicom_dir)
            except zipfile.BadZipFile:
                raise HTTPException(400, "Para tipo=dicom envía un .zip con los .dcm dentro")
            in_path = dicom_dir

        out_obj = os.path.join(work_dir, f"{name_root}.obj")

        # ======= conversión =======
        if segment == "brain":
            # Puedes activar min_voxels para limpiar ruido y ahorrar RAM/caras
            convert_medical_to_obj(in_path, out_obj, is_dicom=(tipo == "dicom"),
                                   threshold=1.0, min_voxels=0)
        else:
            onnx_path = os.getenv("TUMOR_ONNX_PATH")  # None -> usa ./unet3d.onnx
            convert_tumor_to_obj(in_path, out_obj, is_dicom=(tipo == "dicom"),
                                 onnx_model_path=onnx_path)

        if not os.path.isfile(out_obj):
            raise HTTPException(500, "No se generó el archivo OBJ")

        # ======= respuesta + cleanup en background =======
        return FileResponse(
            path=out_obj,
            media_type="application/octet-stream",
            filename=f"{name_root}.obj",
            background=BackgroundTask(lambda: cleanup_paths(work_dir))
        )

    except HTTPException:
        cleanup_paths(work_dir)
        raise
    except Exception as e:
        traceback.print_exc()
        cleanup_paths(work_dir)
        raise HTTPException(500, f"Error en conversión: {e}")
