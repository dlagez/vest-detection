import cv2
import logging

logger = logging.getLogger(__name__)


class CameraPipeline:
    def __init__(self, model_path: str, confidence: float = 0.35, camera_id: int = 0):
        self.model_path = model_path
        self.confidence = confidence
        self.camera_id = camera_id

    def run(self, output_path: str = None, duration: int = None):
        from vest_detection.detector import VestDetector
        from vest_detection.visualizer import DetectionVisualizer

        detector = VestDetector(model_path=self.model_path, confidence=self.confidence)
        visualizer = DetectionVisualizer()

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")

        logger.info(f"摄像头已启动，按 'q' 退出")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("无法读取帧")
                continue

            detections = detector.predict(frame)
            annotated = visualizer.draw(frame, detections)

            cv2.imshow("Vest Detection", annotated)

            if duration and frame_idx >= duration:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"共处理 {frame_idx} 帧")
