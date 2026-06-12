import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestVestDetector(unittest.TestCase):
    """VestDetector 基础测试"""

    @patch("vest_detection.detector.YOLO")
    def test_detector_init(self, mock_yolo):
        """测试检测器初始化"""
        from vest_detection.detector import VestDetector

        detector = VestDetector(
            model_path="weights/best.pt",
            confidence=0.5,
            iou=0.4,
            imgsz=320
        )

        mock_yolo.assert_called_once_with("weights/best.pt")
        self.assertEqual(detector.confidence, 0.5)
        self.assertEqual(detector.iou, 0.4)
        self.assertEqual(detector.imgsz, 320)

    @patch("vest_detection.detector.sv.Detections.from_ultralytics")
    @patch("vest_detection.detector.YOLO")
    def test_predict_returns_detections(self, mock_yolo, mock_from_ultralytics):
        """测试预测返回检测结果"""
        from vest_detection.detector import VestDetector
        import supervision as sv
        import numpy as np

        mock_result = MagicMock()
        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        mock_detections = MagicMock(spec=sv.Detections)
        mock_detections.__len__ = lambda self: 2
        mock_from_ultralytics.return_value = mock_detections

        detector = VestDetector(model_path="weights/best.pt")
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = detector.predict(image)

        self.assertEqual(len(detections), 2)


class TestConfig(unittest.TestCase):
    """配置加载测试"""

    def test_load_config_default(self):
        """测试加载默认配置"""
        from vest_detection.config import load_config

        config = load_config("configs/default.yaml")

        self.assertIn("model", config)
        self.assertIn("classes", config)
        self.assertEqual(config["model"]["confidence"], 0.35)
        self.assertEqual(config["model"]["imgsz"], 640)

    def test_load_config_not_found(self):
        """测试不存在的配置文件"""
        from vest_detection.config import load_config

        with self.assertRaises(FileNotFoundError):
            load_config("configs/nonexistent.yaml")


if __name__ == "__main__":
    unittest.main()
