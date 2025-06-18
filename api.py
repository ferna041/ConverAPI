# api.py
import os
import shutil
import uuid
import zipfile
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

# Importa tu función de conversión
from a import convert_medical_to_obj

app = FastAPI(title="Medical → OBJ Converter")

# Directorio base para temporales
BASE_TMP = "/tmp/convert_api"
os.makedirs(BASE_TMP, exist_ok=True)

def cleanup_paths(*paths):
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
        except:
            pass

@app.post("/convert", summary="Convierte un NIfTI (.nii/.nii.gz) o ZIP DICOM a OBJ")
async def convert_endpoint(
    file: UploadFile = File(...),
    tipo: str = Form("nifti")  # "nifti" o "dicom"
):
    # Validar tipo
    tipo = tipo.lower()
    if tipo not in ("nifti", "dicom"):
        raise HTTPException(400, "tipo debe ser 'nifti' o 'dicom'")

    # Generar IDs únicos y rutas
    job_id    = uuid.uuid4().hex
    upload_fn = file.filename
    name_root = os.path.splitext(upload_fn)[0]
    work_dir  = os.path.join(BASE_TMP, job_id)
    os.makedirs(work_dir, exist_ok=True)

    # Ruta de entrada para la conversión
    if tipo == "nifti":
        # Guardar directamente el .nii/.nii.gz
        in_path  = os.path.join(work_dir, upload_fn)
        with open(in_path, "wb") as f:
            f.write(await file.read())
    else:
        # DICOM: esperamos un ZIP con .dcm dentro
        zip_path = os.path.join(work_dir, f"{name_root}.zip")
        with open(zip_path, "wb") as f:
            f.write(await file.read())
        # Descomprimir ZIP
        dicom_dir = os.path.join(work_dir, "dicom")
        os.makedirs(dicom_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dicom_dir)
        in_path = dicom_dir

    # Ruta de salida .obj
    out_obj = os.path.join(work_dir, f"{name_root}.obj")

    # Llamar a tu función de conversión
    try:
        convert_medical_to_obj(in_path, out_obj, is_dicom=(tipo=="dicom"))
    except Exception as e:
        # Limpieza en caso de fallo
        cleanup_paths(work_dir)
        raise HTTPException(500, f"Error en conversión: {e}")

    # Validar OBJ generado
    if not os.path.isfile(out_obj):
        cleanup_paths(work_dir)
        raise HTTPException(500, "No se generó el archivo OBJ")

    # Devolver el .obj con limpieza posterior
    return FileResponse(
        path=out_obj,
        media_type="application/octet-stream",
        filename=f"{name_root}.obj",
        background=BackgroundTask(lambda: cleanup_paths(work_dir))
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")
