import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
current_id = 0

for person in os.listdir("dataset"):
    label_map[current_id] = person
    for img in os.listdir(f"dataset/{person}"):
        img_path = f"dataset/{person}/{img}"
        image = Image.open(img_path).convert("L")
        faces.append(np.array(image))
        labels.append(current_id)
    current_id += 1

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

with open("labels.txt", "w") as f:
    for k, v in label_map.items():
        f.write(f"{k},{v}\n")

print("Training completed")
