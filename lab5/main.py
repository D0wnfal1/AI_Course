import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = "/kaggle/input/dataset-lab5/train" 
TEST_DIR = "/kaggle/input/dataset-lab5/test"  
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
MODEL_PATH = "inception_like_binary.h5"
CLASS_INDICES_JSON = "class_indices.json"
METADATA_JSON = "metadata.json"

def create_generators(dataset_dir=DATASET_DIR):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.12,
        zoom_range=0.12,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    with open(CLASS_INDICES_JSON, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)
    return train_gen, val_gen

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce, filters_3x3,
                     filters_5x5_reduce, filters_5x5,
                     filters_pool_proj,
                     name=None):
    branch1 = layers.Conv2D(filters_1x1, (1,1), padding='same', activation='relu')(x)
    branch3 = layers.Conv2D(filters_3x3_reduce, (1,1), padding='same', activation='relu')(x)
    branch3 = layers.Conv2D(filters_3x3, (3,3), padding='same', activation='relu')(branch3)
    branch5 = layers.Conv2D(filters_5x5_reduce, (1,1), padding='same', activation='relu')(x)
    branch5 = layers.Conv2D(filters_5x5, (3,3), padding='same', activation='relu')(branch5)
    branch5 = layers.Conv2D(filters_5x5, (3,3), padding='same', activation='relu')(branch5)
    branch_pool = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    branch_pool = layers.Conv2D(filters_pool_proj, (1,1), padding='same', activation='relu')(branch_pool)
    out = layers.concatenate([branch1, branch3, branch5, branch_pool], axis=-1, name=name)
    return out

def build_inception_like(input_shape=(299,299,3), dropout_rate=0.4):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), strides=(2,2), padding='valid', activation='relu')(inp)
    x = layers.Conv2D(32, (3,3), padding='valid', activation='relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='valid')(x)
    x = layers.Conv2D(80, (1,1), activation='relu')(x)
    x = layers.Conv2D(192, (3,3), padding='valid', activation='relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='valid')(x)

    x = inception_module(x, 64, 48, 64, 64, 96, 32, name="incept_1")
    x = inception_module(x, 64, 48, 64, 64, 96, 64, name="incept_2")
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='valid')(x)

    x = inception_module(x, 128, 96, 128, 96, 128, 128, name="incept_3")
    x = inception_module(x, 160, 112, 160, 112, 160, 128, name="incept_4")
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate/2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)  
    model = models.Model(inputs=inp, outputs=out, name="InceptionLike_binary")
    return model

def compile_and_train(model, train_gen, val_gen, epochs=EPOCHS):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    from collections import Counter
    counter = Counter(train_gen.classes)
    majority = max(counter.values())
    class_weight = {cls: float(majority/count) for cls, count in counter.items()}
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=cb,
        verbose=1
    )
    return history

def evaluate_model(model, val_gen):
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1).ravel()
    y_pred = (preds > 0.5).astype(int)
    y_true = val_gen.classes
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()), zero_division=0))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(val_gen.class_indices.keys()), yticklabels=list(val_gen.class_indices.keys()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.show()
    meta = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    with open(METADATA_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "cm":cm}

def predict_image(model, image_path, class_indices_path=CLASS_INDICES_JSON):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = model.predict(arr)[0,0]
    pred = int(prob > 0.5)
    with open(class_indices_path, "r", encoding="utf-8") as f:
        cls = json.load(f)
    inv = {v:k for k,v in cls.items()}
    predicted_label = inv.get(pred, str(pred))
    return predicted_label, float(prob)

if __name__ == "__main__":
    assert os.path.exists(DATASET_DIR), f"DATASET_DIR not found: {DATASET_DIR}"
    train_gen, val_gen = create_generators(DATASET_DIR)
    model = build_inception_like(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dropout_rate=0.4)
    model.summary() 
    history = compile_and_train(model, train_gen, val_gen, epochs=EPOCHS)
    results = evaluate_model(model, val_gen)
    model.save(MODEL_PATH)
    print("Model and metadata saved.")
    if os.path.exists(TEST_DIR):
        print("\n--- Predicting test images ---")
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=1, class_mode=None, shuffle=False)
        preds = model.predict(test_gen, verbose=1)
        pred_labels = (preds.ravel() > 0.5).astype(int)
        inv = {v:k for k,v in train_gen.class_indices.items()}
        for i, fname in enumerate(test_gen.filenames):
            print(f"{fname} -> {inv[pred_labels[i]]} (p={preds.ravel()[i]:.3f})")
    else:
        print("\nℹ️ TEST_DIR not found. If you want to test images, create TEST_DIR")
