from ultralytics import YOLO
import supervision as sv


class VestDetector:
    def __init__(self, model_path: str, confidence: float = 0.35, iou: float = 0.5, imgsz: int = 640):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz

    def predict(self, image, verbose: bool = False):
        result = self.model(
            image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=verbose
        )[0]

        detections = sv.Detections.from_ultralytics(result)
        return detections