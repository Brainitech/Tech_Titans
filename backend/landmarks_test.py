import matplotlib.pyplot as plt

# Define the connections between landmarks (Mediapipe Hand Connections)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky finger
]

def visualize_hand_from_landmarks(landmarks):
    """
    Visualizes hand landmarks on a 2D plane.
    
    Args:
        landmarks (list of tuples): A list of (x, y) coordinates for the 21 hand landmarks.
    """
    if len(landmarks) != 21:
        print("Error: Exactly 21 landmarks are required.")
        return
    
    # Extract X and Y coordinates
    x_coords = [lm[0] for lm in landmarks]
    y_coords = [lm[1] for lm in landmarks]

    # Plot the landmarks
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, color="blue", s=50, label="Landmarks")

    # Draw connections
    for connection in HAND_CONNECTIONS:
        start, end = connection
        plt.plot(
            [x_coords[start], x_coords[end]],
            [y_coords[start], y_coords[end]],
            color="green", linewidth=2
        )
    
    # Adjust plot
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.axis("equal")
    plt.title("Hand Landmark Visualization")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.legend()
    plt.grid(True)
    plt.savefig("hand_landmarks.png")

if __name__ == "__main__":
    # Provided landmarks for a single sign
    raw_landmarks = [
        0.5050510764122009,0.8460731506347656,1.301549104937294e-06,0.5924531817436218,0.7271561622619629,-0.09149827063083649,0.6029492616653442,0.5652662515640259,-0.09635333716869354,0.5044369101524353,0.4515005350112915,-0.09307218343019485,0.4085298776626587,0.36857175827026367,-0.08045614510774612,0.6240592002868652,0.44200897216796875,0.01602248288691044,0.5750204920768738,0.27901673316955566,-0.006571509409695864,0.5275532603263855,0.17681550979614258,-0.02610746957361698,0.48082783818244934,0.08511564135551453,-0.039178505539894104,0.5348608493804932,0.440054714679718,0.03641340509057045,0.5533985495567322,0.25443145632743835,0.0013760578585788608,0.5603172779083252,0.1457878053188324,-0.0260583758354187,0.5656055212020874,0.05505666136741638,-0.034194350242614746,0.44785746932029724,0.47410938143730164,0.03913430497050285,0.39707159996032715,0.35773512721061707,-0.05672048404812813,0.41912245750427246,0.474099338054657,-0.08501958847045898,0.44753363728523254,0.5553440451622009,-0.06692759692668915,0.36562854051589966,0.526781439781189,0.036524295806884766,0.3212486505508423,0.42277464270591736,-0.04359852522611618,0.3476769030094147,0.4894411563873291,-0.05181695520877838,0.37734997272491455,0.5514059662818909,-0.03568004444241524
    ]
    
    # Convert raw landmarks into (x, y) tuples
    landmarks = [(raw_landmarks[i], raw_landmarks[i + 1]) for i in range(0, len(raw_landmarks), 3)]
    
    # Visualize the hand
    visualize_hand_from_landmarks(landmarks)
