import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from vest_detection.pipelines.image_pipeline import ImagePipeline

app = FastAPI(title="Reflective Vest Detection API")

pipeline = ImagePipeline(
    model_path="weights/best.pt",
    confidence=0.35
)


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    input_path = Path("outputs/uploaded.jpg")
    output_path = Path("outputs/images/result.jpg")
    json_path = Path("outputs/json/result.json")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = pipeline.run(
        image_path=str(input_path),
        output_path=str(output_path),
        json_path=str(json_path)
    )

    return {
        "message": "success",
        "result": result,
        "image_url": "/result/image"
    }


@app.get("/result/image")
def get_result_image():
    return FileResponse("outputs/images/result.jpg")