import cv2
import json
import logging
from pathlib import Path

from vest_detection.detector import VestDetector
from vest_detection.visualizer import DetectionVisualizer

logger = logging.getLogger(__name__)


class VideoPipeline:
    def __init__(self, model_path: str, confidence: float = 0.35, enable_tracking: bool = True):
        self.detector = VestDetector(model_path=model_path, confidence=confidence)
        self.visualizer = DetectionVisualizer()
        self.enable_tracking = enable_tracking

    def run(self, video_path: str, output_path: str, json_path: str = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"视频读取失败：{video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.predict(frame, verbose=False)
            annotated = self.visualizer.draw(frame, detections)
            out.write(annotated)

            result_data = self._detections_to_dict(detections, frame_idx)
            all_results.append(result_data)

            frame_idx += 1
            if frame_idx % 30 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()
        out.release()

        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Video saved to {output_path}")
        return all_results

    def _detections_to_dict(self, detections, frame_idx: int):
        class_names = detections.data.get("class_name", [])

        results = []
        for i in range(len(detections)):
            results.append({
                "class_name": str(class_names[i]) if len(class_names) > i else str(detections.class_id[i]),
                "confidence": float(detections.confidence[i]),
                "bbox": detections.xyxy[i].tolist()
            })

        return {
            "frame": frame_idx,
            "count": len(results),
            "detections": results
        }
