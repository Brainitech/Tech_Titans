import os
import cv2
import numpy as np

DATASET_PATH = "DATASET/asl_dataset/"
PROCESSED_PATH = "processed_data"
IMG_SIZE = 256

def preprocess_images():
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
    
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        save_label_path = os.path.join(PROCESSED_PATH, label)
        os.makedirs(save_label_path, exist_ok=True)

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(save_label_path, img_name)
            cv2.imwrite(save_path, img)

if __name__ == "__main__":
    preprocess_images()
