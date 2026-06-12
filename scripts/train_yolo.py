import argparse
import logging

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型训练")
    parser.add_argument("--data", type=str, required=True, help="数据集 YAML 路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="预训练模型")
    parser.add_argument("--output", type=str, default="weights/", help="输出目录")
    args = parser.parse_args()

    logging.info(f"加载模型: {args.model}")
    model = YOLO(args.model)

    logging.info(f"开始训练: data={args.data}, epochs={args.epochs}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.output,
        name="vest_detection"
    )

    logging.info(f"训练完成，最佳权重: {results.save_dir}")


if __name__ == "__main__":
    main()
