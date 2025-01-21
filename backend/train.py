import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

DATASET_PATH = "DATASET/asl_dataset/"
BATCH_SIZE = 32
EPOCHS = 20

def load_data():
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255)

    train_data = datagen.flow_from_directory(DATASET_PATH, target_size=(64, 64),
                                             batch_size=BATCH_SIZE, subset='training')
    val_data = datagen.flow_from_directory(DATASET_PATH, target_size=(64, 64),
                                           batch_size=BATCH_SIZE, subset='validation')
    return train_data, val_data

if __name__ == "__main__":
    train_data, val_data = load_data()
    model = create_model()

    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
    model.save("asl_model.keras")
