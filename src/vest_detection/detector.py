from ultralytics import YOLO
import supervision as sv


class VestDetector:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        class_filter: list | None = None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        # class_filter can be class IDs (int) or class names (str) from the model.
        # YOLO's `classes` param expects integer class IDs.
        self._class_ids = self._resolve_class_filter(class_filter)

    def _resolve_class_filter(self, class_filter):
        if class_filter is None:
            return None
        model_names = self.model.names  # {0: "hat", 1: "nohat", ...}
        ids = []
        for item in class_filter:
            if isinstance(item, int):
                ids.append(item)
            elif isinstance(item, str):
                # Try to match by name
                matched = False
                for cid, cname in model_names.items():
                    if cname.lower() == item.lower():
                        ids.append(cid)
                        matched = True
                        break
                if not matched:
                    raise ValueError(
                        f"Class name '{item}' not found in model. "
                        f"Available: {list(model_names.values())}"
                    )
            else:
                raise TypeError(f"class_filter items must be int or str, got {type(item)}")
        return ids

    def predict(self, image, verbose: bool = False):
        result = self.model(
            image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self._class_ids,
            verbose=verbose,
        )[0]

        detections = sv.Detections.from_ultralytics(result)
        return detections