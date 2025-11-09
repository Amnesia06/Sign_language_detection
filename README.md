
Sign Language Detection Project

This project implements a real-time Sign Language Detection system using hand keypoint data captured via a webcam. The core goal is to interpret a set of signs and provide a foundation for expanding the vocabulary.

The system currently recognizes 9 signs: Thankyou, Welcome, Hello, Bye, Peace, Yes, No, I Love You and Care.




Project Structure

| File/Folder | Purpose |
| :--- | :--- |
| **`app.py`** | The **main file** to run the real-time detection application. |
| **`collectdata.py`** | Script used to **capture raw image frames** from the webcam for new signs. |
| **`data.py`** | Script to **extract numerical keypoint data** from the raw images. |
| **`trainmodel.py`** | Script to **train the neural network** and update the model weights. |
| **`model.json` / `model.h5`** | The architecture and trained weights of the deep learning model. |
| **`Image/`** | Stores **raw image frames** captured for the dataset. |
| **`MP\_Data/`** | Stores the **processed keypoint data** (NumPy arrays) used for model training. |
| **`Logs/`** | Stores TensorBoard logs generated during training. |
| **`Users/`** | Directory for user-specific data (as visualized in the folder structure). |

Getting Started

1.  Dependencies: Install the required Python packages.

    ```bash
    pip install opencv-python mediapipe keras numpy scikit-learn
    ```

2.  Run the App: To see the detection working live with the pre-trained signs, run the main execution file:

    ```bash
    python app.py
    ```


Capturing and Training a New Sign: A Complete Guide

To add a new sign to the system's vocabulary, you need to coordinate the information across the three main Python scripts and then run them in a specific sequence. This process ensures the neural network learns the new gesture and knows what label to give it.

1. Define the New Vocabulary

Before collecting any data, you must manually define the new sign name (e.g., "More") in the vocabulary lists used by the scripts:

* **`collectdata.py`**: Add the new sign name and the key you will press to collect its data (e.g., `'more': 'More'`). This tells the script what folder name to create in **Image/**.
* **`data.py`**: Add the new sign name (e.g., `"More"`) to the `phrases` list so the script knows which new folder to process from **Image/** to **MP\_Data/**.
* **`trainmodel.py`**: Add the new sign name (e.g., `"More"`) to the `phrases` list. This dynamically adds a new output category for the sign in the neural network during training.


2. The 3-Step Data Pipeline

Once the vocabulary lists are updated in all three scripts, follow these steps sequentially:

**Step 1: Collect Raw Images (`collectdata.py`)**

* **Action**: Run `collectdata.py`. When the webcam opens, position your hand for the new sign in the frame. Press the first letter of the new sign's name on the keyboard (e.g., 'm' for "More") repeatedly to save raw images into the **Image/** folder. Press 'q' to stop.

**Step 2: Extract Keypoints (`data.py`)**

* **Action**: Run `data.py`. This script processes those raw images. It uses MediaPipe to translate the hand shape in every image into a 63-coordinate feature vector, saving this numerical data as a NumPy array (`.npy`) in the **MP\_Data/** folder.

**Step 3: Train Model (`trainmodel.py`)**

* **Action**: Run `trainmodel.py`. This script loads *all* the keypoint data (old signs + the new sign's data) from **MP\_Data/** and retrains the neural network. The model's weights are updated to recognize the expanded vocabulary, and the result is saved to `model.h5`.







