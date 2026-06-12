import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml

from vest_detection.pipelines.video_pipeline import VideoPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    default_model = config["model"]["path"]
    default_conf = config["model"]["confidence"]

    parser = argparse.ArgumentParser(description="视频反光背心推理")
    parser.add_argument("--input", type=str, default="data/videos/video.mp4", help="输入视频路径")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径（默认outputs/视频名_时间戳/）")
    parser.add_argument("--json", type=str, default=None, help="输出JSON路径（默认outputs/视频名_时间戳/）")
    parser.add_argument("--model", type=str, default=default_model, help="模型路径")
    parser.add_argument("--conf", type=float, default=default_conf, help="置信度阈值")
    args = parser.parse_args()

    input_path = Path(args.input)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"{input_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output or str(output_dir / f"{input_path.stem}_result.mp4")
    json_path = args.json or str(output_dir / f"{input_path.stem}_result.json")

    pipeline = VideoPipeline(
        model_path=args.model,
        confidence=args.conf
    )

    result = pipeline.run(
        video_path=args.input,
        output_path=output_path,
        json_path=json_path
    )

    logging.info(f"推理完成，共处理 {len(result)} 帧")
    logging.info(f"输出视频: {output_path}")
    logging.info(f"输出JSON: {json_path}")


if __name__ == "__main__":
    main()
