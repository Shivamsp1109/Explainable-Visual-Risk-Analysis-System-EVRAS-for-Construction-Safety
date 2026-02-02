import os
import cv2
import random

CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
]

def draw_yolo_boxes(img, label_path):
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        cv2.putText(img, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img

def main():
    images_dir = "dataset/train/images"
    labels_dir = "dataset/train/labels"

    images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not images:
        print("No images found.")
        return

    sample = random.choice(images)
    img_path = os.path.join(images_dir, sample)
    lbl_path = os.path.join(labels_dir, os.path.splitext(sample)[0] + ".txt")

    img = cv2.imread(img_path)
    img = draw_yolo_boxes(img, lbl_path)

    cv2.imshow("Preview", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()