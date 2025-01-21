import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = "asl_landmark_model.keras"
model = load_model(MODEL_PATH)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Load label encoder
LABELS = [chr(i) for i in range(97, 123)] # a-z, DEL, SPACE

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert landmarks to NumPy array and reshape for the model
            landmarks = np.array(landmarks).reshape(1, -1)  # Shape: (1, 63)

            # Normalize landmarks
            landmarks /= np.max(landmarks)

            # Make prediction
            prediction = model.predict(landmarks)
            predicted_label = LABELS[np.argmax(prediction)]

            # Draw landmarks and prediction on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("ASL Recognition", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
