import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import cv2
from PIL import Image

DATASET_DIR = "/kaggle/input/lab6-dataset/dataset"
WORK_DIR = "/kaggle/working/dataset"
AUGMENTED_DIR = "/kaggle/working/augmented_dataset"
VIDEO_PATH = "/kaggle/input/lab6-video/Nike Air Max 270 - Animation.mp4"
MODEL_PATH = "/kaggle/working/best_model.h5"
TARGET_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS_FROZEN = 15
EPOCHS_FINE_TUNE = 10

print("\n[INFO] Перевірка та очищення датасету...")
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
shutil.copytree(DATASET_DIR, WORK_DIR)
DATASET_DIR = WORK_DIR

corrupted_count = 0
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        try:
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            print(f"[WARNING] Видалено пошкоджене зображення: {file}")
            os.remove(img_path)
            corrupted_count += 1

print(f"[INFO] Датасет очищено. Видалено {corrupted_count} пошкоджених зображень.\n")

print("[INFO] Аналіз розподілу класів...")
positive_count = len(os.listdir(os.path.join(DATASET_DIR, "positive")))
negative_count = len(os.listdir(os.path.join(DATASET_DIR, "negative")))

print(f"Positive клас: {positive_count} зображень")
print(f"Negative клас: {negative_count} зображень")
print(f"Співвідношення: {positive_count / negative_count:.2f}\n")

print("[INFO] Виконується аугментація даних...")
if os.path.exists(AUGMENTED_DIR):
    shutil.rmtree(AUGMENTED_DIR)

os.makedirs(os.path.join(AUGMENTED_DIR, "positive"), exist_ok=True)
os.makedirs(os.path.join(AUGMENTED_DIR, "negative"), exist_ok=True)

shutil.copytree(os.path.join(DATASET_DIR, "positive"),
                os.path.join(AUGMENTED_DIR, "positive"), dirs_exist_ok=True)
shutil.copytree(os.path.join(DATASET_DIR, "negative"),
                os.path.join(AUGMENTED_DIR, "negative"), dirs_exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

if positive_count < negative_count:
    target_class = "positive"
    augment_count = (negative_count - positive_count) // positive_count
else:
    target_class = "negative"
    augment_count = (positive_count - negative_count) // negative_count

augment_count = max(2, min(augment_count, 5))

target_dir = os.path.join(DATASET_DIR, target_class)
output_dir = os.path.join(AUGMENTED_DIR, target_class)

for filename in os.listdir(target_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(target_dir, filename)
    try:
        img = load_img(img_path, target_size=TARGET_SIZE)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        i = 0
        base_name = os.path.splitext(filename)[0]
        for batch in datagen.flow(x, batch_size=1):
            aug_filename = f"{base_name}_aug_{i}.jpg"
            aug_path = os.path.join(output_dir, aug_filename)
            img_aug = tf.keras.preprocessing.image.array_to_img(batch[0])
            img_aug.save(aug_path)
            i += 1
            if i >= augment_count:
                break
    except Exception as e:
        print(f"[WARNING] Помилка аугментації {filename}: {e}")

DATASET_DIR = AUGMENTED_DIR

positive_count_aug = len(os.listdir(os.path.join(DATASET_DIR, "positive")))
negative_count_aug = len(os.listdir(os.path.join(DATASET_DIR, "negative")))
print(f"[INFO] Після аугментації:")
print(f"Positive клас: {positive_count_aug} зображень")
print(f"Negative клас: {negative_count_aug} зображень\n")

print("[INFO] Підготовка даних...")

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Тренувальних зображень: {train_generator.samples}")
print(f"Валідаційних зображень: {validation_generator.samples}\n")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"[INFO] Ваги класів: {class_weight_dict}\n")

print("[INFO] Створення моделі Xception...")
input_shape = TARGET_SIZE + (3,)
input_tensor = Input(shape=input_shape)

base_model = Xception(
    include_top=False,
    weights='imagenet',
    input_tensor=input_tensor
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print(f"[INFO] Всього параметрів: {model.count_params():,}")
print(f"[INFO] Тренувальних параметрів: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}\n")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n" + "=" * 60)
print("[INFO] ФАЗА 1: Навчання з замороженою базовою моделлю")
print("=" * 60 + "\n")

history_frozen = model.fit(
    train_generator,
    epochs=EPOCHS_FROZEN,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 60)
print("[INFO] ФАЗА 2: Fine-tuning (розморожування останніх шарів)")
print("=" * 60 + "\n")

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print(
    f"[INFO] Тренувальних параметрів після розморожування: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}\n")

history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=validation_generator,
    initial_epoch=len(history_frozen.history['loss']),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

history_combined = {}
for key in history_frozen.history.keys():
    history_combined[key] = history_frozen.history[key] + history_fine.history[key]

print("\n[INFO] Візуалізація результатів...")
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.plot(history_combined['accuracy'], label="Training Accuracy", linewidth=2)
plt.plot(history_combined['val_accuracy'], label="Validation Accuracy", linewidth=2)
plt.axvline(x=EPOCHS_FROZEN, color='red', linestyle='--', label='Fine-tuning Start')
plt.title("Accuracy over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history_combined['loss'], label="Training Loss", linewidth=2)
plt.plot(history_combined['val_loss'], label="Validation Loss", linewidth=2)
plt.axvline(x=EPOCHS_FROZEN, color='red', linestyle='--', label='Fine-tuning Start')
plt.title("Loss over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(history_combined['auc'], label="Training AUC", linewidth=2)
plt.plot(history_combined['val_auc'], label="Validation AUC", linewidth=2)
plt.axvline(x=EPOCHS_FROZEN, color='red', linestyle='--', label='Fine-tuning Start')
plt.title("AUC over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("[INFO] ОЦІНКА МОДЕЛІ НА ВАЛІДАЦІЙНОМУ НАБОРІ")
print("=" * 60 + "\n")

model.load_weights(MODEL_PATH)

Y_pred = model.predict(validation_generator, verbose=1)
y_pred = (Y_pred > 0.5).astype(int).flatten()

cm = confusion_matrix(validation_generator.classes, y_pred)
print("\nМатриця помилок:")
print(cm)

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'=' * 50}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")
print(f"{'=' * 50}\n")

print("\nClassification Report:")
print(classification_report(
    validation_generator.classes,
    y_pred,
    target_names=['Negative', 'Positive']
))

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    cbar_kws={'label': 'Count'}
)
plt.ylabel("Actual", fontsize=12, fontweight='bold')
plt.xlabel("Predicted", fontsize=12, fontweight='bold')
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n[INFO] Аналіз випадкових зображень з валідаційного набору...")
val_images, val_labels = [], []
for i in range(len(validation_generator.filenames)):
    img_path = os.path.join(AUGMENTED_DIR, validation_generator.filenames[i])
    val_images.append(img_path)
    val_labels.append(validation_generator.classes[i])

samples = random.sample(list(zip(val_images, val_labels)), min(10, len(val_images)))

plt.figure(figsize=(20, 8))
for i, (img_path, true_label) in enumerate(samples):
    img = load_img(img_path, target_size=TARGET_SIZE)
    img_array = img_to_array(img) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0]
    pred_label = 1 if prediction > 0.5 else 0

    is_correct = pred_label == true_label
    border_color = 'green' if is_correct else 'red'

    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")

    title_text = f"True: {'Positive' if true_label == 1 else 'Negative'}\n"
    title_text += f"Pred: {'Positive' if pred_label == 1 else 'Negative'}\n"
    title_text += f"Conf: {prediction:.3f}"

    plt.title(title_text, color=border_color, fontweight='bold')

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)

plt.tight_layout()
plt.savefig('/kaggle/working/sample_predictions.png', dpi=300, bbox_inches='tight')
plt.show()


def smooth_predictions(predictions, window_size=7, threshold=0.5):
    """Постобробка: згладжування результатів передбачень."""
    smoothed = []
    for i in range(len(predictions)):
        window = predictions[max(0, i - window_size // 2):i + window_size // 2 + 1]
        avg = np.mean(window)
        smoothed.append(1 if avg > threshold else 0)
    return smoothed


def process_video_frame_by_frame(video_path, model, target_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Не вдалося відкрити відео!")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_predictions = []
    frame_times = []

    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        time_sec = frame_count / fps
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img_array = img_to_array(img) / 255.0
        pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0]

        frame_predictions.append(pred)
        frame_times.append(time_sec)

    cap.release()
    total_time = time.time() - start_time
    return frame_predictions, total_time


def process_video_batch(video_path, model, target_size, batch_size=32, skip_frames=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Не вдалося відкрити відео!")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_predictions = []
    frame_times = []

    frame_batch = []
    batch_frame_times = []
    frame_count = 0

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        time_sec = frame_count / fps
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        frame_batch.append(img / 255.0)
        batch_frame_times.append(time_sec)

        if len(frame_batch) == batch_size:
            predictions = model.predict(np.array(frame_batch), verbose=0)
            frame_predictions.extend(pred[0] for pred in predictions)
            frame_times.extend(batch_frame_times)
            frame_batch, batch_frame_times = [], []

    if len(frame_batch) > 0:
        predictions = model.predict(np.array(frame_batch), verbose=0)
        frame_predictions.extend(pred[0] for pred in predictions)
        frame_times.extend(batch_frame_times)

    cap.release()
    total_time = time.time() - start_time
    return frame_predictions, total_time


print("\n" + "=" * 60)
print("[INFO] ДОСЛІДЖЕННЯ ШВИДКОСТІ ТА ТОЧНОСТІ ОБРОБКИ ВІДЕО")
print("=" * 60 + "\n")

frame_preds_frame_by_frame, time_frame_by_frame = process_video_frame_by_frame(VIDEO_PATH, model, TARGET_SIZE)

frame_preds_batch, time_batch = process_video_batch(VIDEO_PATH, model, TARGET_SIZE, batch_size=32, skip_frames=3)

smoothed_batch_preds = smooth_predictions(frame_preds_batch, window_size=7, threshold=0.5)

print(f"Час обробки (кадр за кадром): {time_frame_by_frame:.2f} сек")
print(f"Час обробки (батчами): {time_batch:.2f} сек")
print(f"Прискорення: {time_frame_by_frame / time_batch:.2f}x")

plt.figure(figsize=(14, 6))
plt.plot(frame_preds_batch, label="Оригінальні передбачення (батчі)", alpha=0.7)
plt.plot(smoothed_batch_preds, label="Постоброблені передбачення", linewidth=2)
plt.title("Порівняння передбачень з постобробкою та без неї")
plt.xlabel("Кадри")
plt.ylabel("Ймовірність")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\n[INFO] Дослідження завершено.")
print(f"Час обробки відео кадр за кадром: {time_frame_by_frame:.2f} сек")
print(f"Час обробки відео батчами: {time_batch:.2f} сек")
print(f"Прискорення: {time_frame_by_frame / time_batch:.2f}x")