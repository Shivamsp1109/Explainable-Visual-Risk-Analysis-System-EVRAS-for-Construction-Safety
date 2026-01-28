from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path: str):
        results = self.model(image_path)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": results.names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        return detections