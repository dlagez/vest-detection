import supervision as sv


class DetectionVisualizer:
    def __init__(self, display_names: dict | None = None):
        """
        Args:
            display_names: Mapping from class_name -> display label.
                           e.g. {"hat": "安全帽", "vest": "穿戴反光衣"}
        """
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.display_names = display_names or {}

    def draw(self, image, detections):
        labels = []

        class_names = detections.data.get("class_name", None)

        if class_names is not None:
            for class_name, confidence in zip(class_names, detections.confidence):
                display = self.display_names.get(class_name, class_name)
                labels.append(f"{display} {confidence:.2f}")
        else:
            for class_id, confidence in zip(detections.class_id, detections.confidence):
                labels.append(f"{class_id} {confidence:.2f}")

        annotated = self.box_annotator.annotate(
            scene=image.copy(),
            detections=detections
        )

        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels
        )

        return annotated