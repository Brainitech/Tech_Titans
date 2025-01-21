import cv2
import mediapipe as mp
import os
import csv

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# Paths
DATASET_DIR = "DATASET2/asl_alphabet_train/"  # Update this path
CSV_FILE = "landmarks.csv"

# Create or open CSV file
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]+["label"]
    writer.writerow(header)

    # Process each folder (label)
    valid_extensions = (".jpg", ".jpeg", ".png")
    for label in sorted(os.listdir(DATASET_DIR)):
        label_path = os.path.join(DATASET_DIR, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                if not img_name.lower().endswith(valid_extensions):
                    print(f"Skipped {img_name}: invalid file format.")
                    continue

                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Skipped {img_path}: could not read the image.")
                    continue

                # Convert the image to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize the image for better detection
                image_rgb = cv2.resize(image_rgb, (256, 256))
                    # Process the image with Mediapipe
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract landmarks
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        # Write landmarks to CSV
                        writer.writerow(landmarks +[label])
                        print(f"Hand detected and landmarks saved for {img_path}")
                else:
                    print(f"No hands detected yet in {img_path}. Retrying...")

print("Landmark extraction complete! Data saved to", CSV_FILE)
