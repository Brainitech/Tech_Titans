import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
CSV_PATH = "landmarks.csv"  # Path to the CSV file
MODEL_SAVE_PATH = "asl_landmark_model.keras"

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Load and preprocess data
def load_data(csv_path):
    # Load CSV
    data = pd.read_csv(csv_path)

    # Split features (landmarks) and labels
    X = data.iloc[:, :-1].values  # All columns except the last
    y = data.iloc[:, -1].values   # The last column (labels)

    # Normalize landmarks (optional, as Mediapipe already scales them)
    # X = X / np.max(np.abs(X), axis=0)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

# Define the model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training and evaluation
def train_and_evaluate(csv_path, model_save_path):
    # Load data
    X_train, X_test, y_train, y_test, label_encoder = load_data(csv_path)

    # Get input shape and number of classes
    input_shape = X_train.shape[1]  # Number of features
    num_classes = len(label_encoder.classes_)  # Number of unique labels

    # Create model
    model = create_model((input_shape,), num_classes)

    # Train model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save the model and label encoder classes
    model.save(model_save_path)
    np.save("label_classes.npy", label_encoder.classes_)
    print(f"Model saved to {model_save_path} and label classes saved to label_classes.npy")

if __name__ == "__main__":
    train_and_evaluate(CSV_PATH, MODEL_SAVE_PATH)
