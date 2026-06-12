from vest_detection.pipelines.image_pipeline import ImagePipeline


def main():
    pipeline = ImagePipeline(
        model_path="weights/best.pt",
        confidence=0.35
    )

    result = pipeline.run(
        image_path="data/images/test.jpg",
        output_path="outputs/images/result.jpg",
        json_path="outputs/json/result.json"
    )

    print(result)


if __name__ == "__main__":
    main()