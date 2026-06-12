import supervision as sv


class DetectionVisualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def draw(self, image, detections):
        labels = []

        class_names = detections.data.get("class_name", None)

        if class_names is not None:
            for class_name, confidence in zip(class_names, detections.confidence):
                labels.append(f"{class_name} {confidence:.2f}")
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