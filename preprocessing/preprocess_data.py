import os
import cv2
import numpy as np

DATASET_PATH = "../dataset"

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

print("Images Shape:", X.shape)
print("Labels Shape:", y.shape)

print("Tumor Samples:", np.sum(y == 1))
print("Normal Samples:", np.sum(y == 0))