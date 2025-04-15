import cv2
import os
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function for MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

# Function for extracting keypoints
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Assuming only one hand is detected
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
        return rh
    else:
        return np.zeros(21*3)

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Phrases to capture
phrases = ["Thankyou", "Welcome", "Hello", "Bye", "Peace", "Yes", "No", "ILoveYou", "Care"]

no_sequences = 100
sequence_length = 100

# Create directories for each phrase if they don't exist
for phrase in phrases:
    phrase_path = os.path.join(DATA_PATH, phrase)
    os.makedirs(phrase_path, exist_ok=True)

# Set up camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for phrase in phrases:
        phrase_folder = os.path.join('Image', phrase)
        for frame_num in range(sequence_length):
            # Read saved image from "Image" folder
            image_path = os.path.join(phrase_folder, f'{frame_num}.png')
            if not os.path.isfile(image_path):
                print(f"Error: Image {image_path} not found")
                continue
            image = cv2.imread(image_path)
            
            # Make detections
            image, results = mediapipe_detection(image, hands)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            
            # Save keypoints
            npy_path = os.path.join(DATA_PATH, phrase, str(frame_num) + '.npy')
            np.save(npy_path, keypoints)
            print(f"Saved keypoints for frame {frame_num} for phrase {phrase}")

            # Display keypoints on the image (for visualization)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)
            
            # Show the image with keypoints (for visualization)
            cv2.imshow("Hand Landmarks", image)
            cv2.waitKey(200)

cv2.destroyAllWindows()
