import cv2
import json
from pathlib import Path

from vest_detection.detector import VestDetector
from vest_detection.visualizer import DetectionVisualizer


class ImagePipeline:
    def __init__(self, model_path: str, confidence: float = 0.35):
        self.detector = VestDetector(model_path=model_path, confidence=confidence)
        self.visualizer = DetectionVisualizer()

    def run(self, image_path: str, output_path: str, json_path: str = None):
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"图片读取失败：{image_path}")

        detections = self.detector.predict(image, verbose=False)
        annotated = self.visualizer.draw(image, detections)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, annotated)

        result_data = self._detections_to_dict(detections)

        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

        return result_data

    def _detections_to_dict(self, detections):
        class_names = detections.data.get("class_name", [])

        results = []
        for i in range(len(detections)):
            results.append({
                "class_name": str(class_names[i]) if len(class_names) > i else str(detections.class_id[i]),
                "confidence": float(detections.confidence[i]),
                "bbox": detections.xyxy[i].tolist()
            })

        return {
            "count": len(results),
            "detections": results
        }