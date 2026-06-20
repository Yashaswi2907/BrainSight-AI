import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import os


# ==========================t
# PATHS
# ==========================

TRAIN_DIR = "dataset_v2/Training"
TEST_DIR = "dataset_v2/Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================
# DATA AUGMENTATION
# ==========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    brightness_range=[0.8,1.2]
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================
# DATA LOADERS
# ==========================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)
print("\nCLASS INDICES:")
print(train_generator.class_indices)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==========================
# MODEL
# ==========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    BatchNormalization(),

    Dropout(0.4),

    Dense(256, activation="relu"),

    BatchNormalization(),

    Dropout(0.3),

    Dense(4, activation="softmax")
])

# ==========================
# COMPILE
# ==========================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# CALLBACKS
# ==========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/brain_tumor_multiclass.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# ==========================
# TRAIN
# ==========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[
        early_stop,
        checkpoint
    ]
)

# ==========================
# EVALUATION
# ==========================

loss, accuracy = model.evaluate(test_generator)

print("\nTest Accuracy:", accuracy)

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)

y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_generator.class_indices.keys()
)

plt.figure(figsize=(8,8))

disp.plot(cmap="Blues")

os.makedirs("results", exist_ok=True)

plt.savefig("results/confusion_matrix.png")

plt.show()

# ==========================
# CONFUSION MATRIX
# ==========================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)

print("\n========== CONFUSION MATRIX ==========\n")

print(
    confusion_matrix(
        test_generator.classes,
        y_pred
    )
)

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        test_generator.classes,
        y_pred,
        target_names=list(test_generator.class_indices.keys())
    )
)

# ==========================
# SAVE FINAL MODEL
# ==========================

os.makedirs("models", exist_ok=True)

model.save("models/brain_tumor_multiclass.keras")

print("\nModel Saved Successfully!")

# ==========================
# PLOTS
# ==========================

plt.figure(figsize=(10,5))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")

plt.legend([
    "Train",
    "Validation"
])

plt.show()