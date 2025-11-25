import random
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
import yaml
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("/kaggle/input/dataset-yolo-mkr")
OUTPUT_PATH = Path("/kaggle/working/dataset-augmentation")

NUM_AUG = 7
SEED = 42
MIN_AREA_THRESHOLD = 0.0005
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TRAIN_RATIO = 0.8

random.seed(SEED)
np.random.seed(SEED)

images_folder = INPUT_PATH / "images"
labels_folder = INPUT_PATH / "labels"

if not images_folder.exists():
    raise FileNotFoundError(f"Missing images folder: {images_folder}")
if not labels_folder.exists():
    raise FileNotFoundError(f"Missing labels folder: {labels_folder}")

for subset in ["train", "val"]:
    (OUTPUT_PATH / subset / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / subset / "labels").mkdir(parents=True, exist_ok=True)

augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=25,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.GaussNoise(p=0.3),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
], bbox_params=A.BboxParams(format="yolo", label_fields=["labels"], min_area=1, min_visibility=0.3))

def read_labels(file_path):
    boxes = []
    classes = []
    if not file_path.is_file():
        return boxes, classes
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 5:
                continue
            cls_id, cx, cy, w, h = map(float, tokens)
            boxes.append([cx, cy, w, h])
            classes.append(int(cls_id))
    return boxes, classes

def write_labels(file_path, boxes, classes):
    with open(file_path, "w") as f:
        for cls_id, box in zip(classes, boxes):
            cx, cy, w, h = box
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def bbox_valid(box):
    x, y, w, h = box
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return False
    if not (0 < w <= 1 and 0 < h <= 1):
        return False
    if (x - w / 2) < 0 or (x + w / 2) > 1:
        return False
    if (y - h / 2) < 0 or (y + h / 2) > 1:
        return False
    return True

def bbox_size(box):
    return box[2] * box[3]

image_list = [f for f in images_folder.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
print(f"Discovered {len(image_list)} images.")

train_imgs, val_imgs = train_test_split(image_list, train_size=TRAIN_RATIO, random_state=SEED)
print(f"Split into train: {len(train_imgs)}, val: {len(val_imgs)}")
print("Augmentation starting...")

def process_images(files, partition, do_augment=True):
    saved_images = 0
    out_images_dir = OUTPUT_PATH / partition / "images"
    out_labels_dir = OUTPUT_PATH / partition / "labels"
    for image_path in files:
        filename = image_path.stem
        label_path = labels_folder / f"{filename}.txt"

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load {image_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes, classes = read_labels(label_path)
        if not bboxes:
            print(f"Skipping {filename} due to no bounding boxes")
            continue

        cv2.imwrite(str(out_images_dir / f"{filename}.jpg"), img)
        write_labels(out_labels_dir / f"{filename}.txt", bboxes, classes)
        saved_images += 1

        if do_augment and partition == "train":
            for i in range(NUM_AUG):
                try:
                    augmented = augmentations(image=img_rgb, bboxes=bboxes, labels=classes)
                    aug_img = augmented["image"]
                    aug_boxes = augmented["bboxes"]
                    aug_classes = augmented["labels"]

                    filtered_boxes = []
                    filtered_classes = []
                    for box, cls in zip(aug_boxes, aug_classes):
                        if bbox_valid(box) and bbox_size(box) >= MIN_AREA_THRESHOLD:
                            filtered_boxes.append(box)
                            filtered_classes.append(cls)

                    if not filtered_boxes:
                        continue

                    out_name = f"{filename}_aug{i}.jpg"
                    cv2.imwrite(str(out_images_dir / out_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    write_labels(out_labels_dir / f"{filename}_aug{i}.txt", filtered_boxes, filtered_classes)
                    saved_images += 1
                except Exception as e:
                    print(f"Augmentation failed on {filename}_aug{i}: {e}")
                    continue
    return saved_images

train_saved = process_images(train_imgs, "train", True)
print(f"Train set saved images: {train_saved}")

val_saved = process_images(val_imgs, "val", False)
print(f"Validation set saved images: {val_saved}")

print(f"Overall images saved: {train_saved + val_saved}")

all_class_ids = set()
for part in ["train", "val"]:
    labels_dir = OUTPUT_PATH / part / "labels"
    for lbl_file in labels_dir.glob("*.txt"):
        with open(lbl_file, "r") as f:
            for line in f:
                class_id = int(float(line.split()[0]))
                all_class_ids.add(class_id)

sorted_names = [f"class{cls}" for cls in sorted(all_class_ids)]

yaml_data = {
    "path": str(OUTPUT_PATH.resolve()),
    "train": "train/images",
    "val": "val/images",
    "nc": len(sorted_names),
    "names": sorted_names
}

with open(OUTPUT_PATH / "data.yaml", "w") as yf:
    yaml.dump(yaml_data, yf, default_flow_style=False)

print(f"\nYAML created with {len(sorted_names)} classes")
print(f"Dataset layout:\n{OUTPUT_PATH}/\n  train/\n    images/\n    labels/\n  val/\n    images/\n    labels/\n  data.yaml")