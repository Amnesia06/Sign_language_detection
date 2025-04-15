import os
import cv2

# Set up the webcam
cap = cv2.VideoCapture(0)
directory = 'Image/'

# Define the phrases and their corresponding directories
phrases = {
    'thankyou': 'Thankyou', 
    'bye': 'Bye', 
    'hello': 'Hello', 
    'iloveyou': 'ILoveYou', 
    'no': 'No', 
    'peace': 'Peace', 
    'welcome': 'Welcome', 
    'yes': 'Yes',
    'care':'Care'
}

# Create directories for each phrase if they don't exist
for phrase_dir in phrases.values():
    os.makedirs(os.path.join(directory, phrase_dir), exist_ok=True)

while True:
    _, frame = cap.read()

    # Display the frame
    cv2.imshow("Data", frame)

    # Crop the ROI (region of interest)
    roi = frame[40:400, 0:300]

    # Display the ROI
    cv2.imshow("ROI", roi)

    # Capture key press events
    key = cv2.waitKey(1) & 0xFF

    # Check if the key pressed corresponds to any phrase
    for phrase, phrase_dir in phrases.items():
        if key == ord(phrase[0]):  # Use the first letter of the phrase as the key
            # Count the number of images already captured for this phrase
            count = len(os.listdir(os.path.join(directory, phrase_dir)))
            # Save the ROI as an image in the corresponding directory
            cv2.imwrite(os.path.join(directory, phrase_dir, f"{count}.png"), roi)
            print(f"Saved image for {phrase} as {count}.png")

    # Quit the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
