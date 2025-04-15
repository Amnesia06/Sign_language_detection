import cv2
import numpy as np
from keras.models import model_from_json
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model
model = model_from_json(open("model.json", "r").read())
model.load_weights("model.h5")

phrases = ["Thankyou", "Welcome", "Hello", "Bye", "Peace", "Yes", "No", "ILoveYou", "Care"]
threshold = 0.3  # Lowered for testing

def extract_keypoints(results):
    if not results.multi_hand_landmarks:
        return np.zeros(63)  # Match model input shape
    return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        crop_frame = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
        
        # MediaPipe detection
        image = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Prediction
            keypoints = extract_keypoints(results)
            res = model.predict(np.expand_dims(keypoints, axis=0))[0]
            prediction = np.argmax(res)
            
            if res[prediction] > threshold:
                cv2.putText(frame, phrases[prediction], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()