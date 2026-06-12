import argparse
import logging

from vest_detection.pipelines.video_pipeline import VideoPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="视频反光背心推理")
    parser.add_argument("--input", type=str, default="outputs/videos/video.mp4", help="输入视频路径")
    parser.add_argument("--output", type=str, default="outputs/videos/result.mp4", help="输出视频路径")
    parser.add_argument("--json", type=str, default="outputs/json/result.json", help="输出JSON路径")
    parser.add_argument("--model", type=str, default="weights/best.pt", help="模型路径")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    args = parser.parse_args()

    pipeline = VideoPipeline(
        model_path=args.model,
        confidence=args.conf
    )

    result = pipeline.run(
        video_path=args.input,
        output_path=args.output,
        json_path=args.json
    )

    logging.info(f"推理完成，共处理 {len(result)} 帧")


if __name__ == "__main__":
    main()
