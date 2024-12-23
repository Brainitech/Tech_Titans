#Importing Libraries
import cv2
import mediapipe as mp

#Initialising the Hand Tracking Module
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

#Initialising the Camera
cap = cv2.VideoCapture(0)
hands = mphands.Hands()

#Running the Hand Tracking Module
while True:
    data,image=cap.read() 
    image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB) #Flip the image
    results = hands.process(image) #Process the image

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Convert the image back to BGR

    #Drawing the landmarks on the hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)
    
    #Display the image
    cv2.imshow('Hand Tracking', image)
    cv2.waitKey(1)

    #Exit the loop
    if cv2.waitKey(5) & 0xFF == 27:
        break
