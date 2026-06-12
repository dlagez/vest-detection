import argparse
import logging

from vest_detection.pipelines.camera_pipeline import CameraPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="摄像头实时反光背心检测")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备ID")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="模型路径")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--duration", type=int, default=None, help="最大帧数（可选）")
    args = parser.parse_args()

    pipeline = CameraPipeline(
        model_path=args.model,
        confidence=args.conf,
        camera_id=args.camera
    )

    pipeline.run(duration=args.duration)


if __name__ == "__main__":
    main()
