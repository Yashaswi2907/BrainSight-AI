import os
import random
import matplotlib.pyplot as plt
import cv2

yes_path = "../dataset/yes"
no_path = "../dataset/no"

yes_image = random.choice(os.listdir(yes_path))
no_image = random.choice(os.listdir(no_path))

yes_img = cv2.imread(os.path.join(yes_path, yes_image))
no_img = cv2.imread(os.path.join(no_path, no_image))

yes_img = cv2.cvtColor(yes_img, cv2.COLOR_BGR2RGB)
no_img = cv2.cvtColor(no_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(yes_img)
plt.title("Tumor MRI")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(no_img)
plt.title("Normal MRI")
plt.axis("off")

plt.show()