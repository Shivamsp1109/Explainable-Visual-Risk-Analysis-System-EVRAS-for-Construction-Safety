from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="weights/ppe_yolov8s_best.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path: str, conf_thresh: float = 0.15):
        results = self.model(image_path, conf=conf_thresh, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": results.names.get(cls_id, str(cls_id)),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return detections
