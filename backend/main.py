import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Initialize MediaPipe Hand Module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Path to your dataset
dataset_path = "path/to/dataset"
data = []
labels = []

# Preprocess dataset
print("Processing dataset...")
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue
    for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        if result.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for lm in result.multi_hand_landmarks[0].landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            data.append(np.array(landmarks).flatten())
            labels.append(label)

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a model (Random Forest Classifier)
print("Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, "asl_model.pkl")
print("Model saved as 'asl_model.pkl'")

# Real-time prediction
print("Starting real-time prediction...")
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

model = joblib.load("asl_model.pkl")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            prediction = model.predict([landmarks])[0]
            cv2.putText(image, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ASL Translator", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
