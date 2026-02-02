import os
import shutil

# Old dataset path
SRC_DATASET = r"D:\EVRAS\dataset"

# New cleaned dataset path
DST_DATASET = r"D:\EVRAS\dataset_clean"

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# Old -> New class mapping
CLASS_MAP = {
    5: 0,  # Person -> Person
    0: 1,  # Hardhat -> Hardhat
    1: 2,  # Mask -> Mask
    7: 3,  # Safety Vest -> Safety Vest
    6: 4,  # Safety Cone -> Safety Cone
    8: 5,  # machinery -> machinery
    9: 6,  # vehicle -> vehicle
}

# Classes to ignore
IGNORE_CLASSES = {2, 3, 4}  # NO-Hardhat, NO-Mask, NO-Safety Vest


def ensure_dirs(split):
    os.makedirs(os.path.join(DST_DATASET, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DST_DATASET, split, "labels"), exist_ok=True)


def remap_label_file(src_label_path, dst_label_path):
    if not os.path.exists(src_label_path):
        # If label doesn't exist, create empty file
        open(dst_label_path, "w").close()
        return

    new_lines = []

    with open(src_label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        old_cls = int(parts[0])
        bbox = parts[1:]

        if old_cls in IGNORE_CLASSES:
            continue

        if old_cls not in CLASS_MAP:
            continue

        new_cls = CLASS_MAP[old_cls]
        new_lines.append(" ".join([str(new_cls)] + bbox))

    with open(dst_label_path, "w") as f:
        f.write("\n".join(new_lines))


def process_split(split):
    ensure_dirs(split)

    src_images_dir = os.path.join(SRC_DATASET, split, "images")
    src_labels_dir = os.path.join(SRC_DATASET, split, "labels")

    dst_images_dir = os.path.join(DST_DATASET, split, "images")
    dst_labels_dir = os.path.join(DST_DATASET, split, "labels")

    image_files = [
        f for f in os.listdir(src_images_dir)
        if f.lower().endswith(VALID_IMAGE_EXTS)
    ]

    for img_name in image_files:
        src_img_path = os.path.join(src_images_dir, img_name)
        dst_img_path = os.path.join(dst_images_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        src_lbl_path = os.path.join(src_labels_dir, base_name + ".txt")
        dst_lbl_path = os.path.join(dst_labels_dir, base_name + ".txt")

        shutil.copy2(src_img_path, dst_img_path)
        remap_label_file(src_lbl_path, dst_lbl_path)

    print(f"Done split: {split}, images: {len(image_files)}")


def main():
    for split in ["train", "val", "test"]:
        process_split(split)

    print("\nDataset remapped successfully.")
    print("New dataset created at:", DST_DATASET)


if __name__ == "__main__":
    main()