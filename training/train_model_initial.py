import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# DATA LOADING
# ==========================

DATASET_PATH = "dataset"

categories = ["yes", "no"]

X = []
y = []

IMG_SIZE = 224

for category in categories:

    path = os.path.join(DATASET_PATH, category)

    label = 1 if category == "yes" else 0

    for img in os.listdir(path):

        try:

            img_path = os.path.join(path, img)

            image = cv2.imread(img_path)

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            X.append(image)

            y.append(label)

        except:
            pass

X = np.array(X)
y = np.array(y)

# ==========================
# NORMALIZATION
# ==========================

X = X / 255.0

# ==========================
# SPLIT
# ==========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# ==========================
# MOBILENETV2
# ==========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ==========================
# MODEL
# ==========================

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.3),

    Dense(128, activation="relu"),

    Dense(1, activation="sigmoid")
])

# ==========================
# COMPILE
# ==========================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# EARLY STOPPING
# ==========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==========================
# TRAIN
# ==========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=16,
    callbacks=[early_stop]
)

# ==========================
# SAVE MODEL
# ==========================

os.makedirs("../models", exist_ok=True)

model.save("models/brain_tumor_model.h5")

print("\nModel Saved Successfully!")