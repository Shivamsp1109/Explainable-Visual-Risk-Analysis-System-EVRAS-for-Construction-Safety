import os

def count_files(folder, exts=None):
    if not os.path.exists(folder):
        return 0
    files = os.listdir(folder)
    if exts:
        files = [f for f in files if f.lower().endswith(exts)]
    return len(files)

def check_split(split_name):
    images_dir = f"dataset/{split_name}/images"
    labels_dir = f"dataset/{split_name}/labels"

    img_count = count_files(images_dir, (".jpg", ".jpeg", ".png"))
    lbl_count = count_files(labels_dir, (".txt",))

    print(f"{split_name.upper()} -> images: {img_count}, labels: {lbl_count}")

    if img_count != lbl_count:
        print(f"WARNING: Count mismatch in {split_name}")

def main():
    check_split("train")
    check_split("val")
    check_split("test")

if __name__ == "__main__":
    main()