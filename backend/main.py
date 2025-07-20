import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil

from backend_ml.model.predict import predict


app = FastAPI()

UPLOAD_DIR = "resource/images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict(file_path)
    if not prediction:
        return JSONResponse(status_code=500, content={"error": "Prediction failed"})
    return JSONResponse(content={"prediction": prediction})