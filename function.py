import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Function for MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

# Function for drawing styled landmarks
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

# Function for extracting keypoints
def extract_keypoints(results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
        return np.concatenate([rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Phrases to capture
phrases = np.array(["Thankyou", "Welcome", "Hello", "Bye", "Peace", "Yes", "No", "ILoveYou", "Care"])

no_sequences = 100
sequence_length = 100

# Create directories for each phrase if they don't exist
for phrase in phrases:
    phrase_path = os.path.join(DATA_PATH, phrase)
    os.makedirs(phrase_path, exist_ok=True)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1) # Flip horizontally for a mirrored image
        image, results = mediapipe_detection(frame, hands)

        draw_styled_landmarks(image, results)

        cv2.imshow("MediaPipe Hand Detection", image)

        interrupt = cv2.waitKey(10)
        for phrase in phrases:
            if interrupt & 0xFF == ord(phrase.lower()[0]):
                # Capture sequence of images for each phrase
                for sequence in range(no_sequences):
                    for frame_num in range(sequence_length):
                        _, frame = cap.read()
                        frame = cv2.flip(frame, 1) # Flip horizontally for a mirrored image
                        image, results = mediapipe_detection(frame, hands)
                        draw_styled_landmarks(image, results)
                        cv2.imshow("MediaPipe Hand Detection", image)
                        keypoints = extract_keypoints(results)
                        if frame_num == 0:
                            keypoints_sequence = keypoints[np.newaxis, :]
                        else:
                            keypoints_sequence = np.concatenate((keypoints_sequence, keypoints[np.newaxis, :]), axis=0)
                    
                    np.save(os.path.join(DATA_PATH, phrase, str(sequence + 1)), keypoints_sequence)
                    print(f"Saved sequence {sequence + 1} for phrase {phrase}")

        if interrupt & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
