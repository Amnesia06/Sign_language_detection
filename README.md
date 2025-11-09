
Sign Language Detection Project

This project implements a real-time Sign Language Detection system using hand keypoint data captured via a webcam. The core goal is to interpret a set of signs and provide a foundation for expanding the vocabulary.

The system currently recognizes 9 signs: Thankyou, Welcome, Hello, Bye, Peace, Yes, No, I Love You (ILoveYou), and Care.




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

Capturing New Signs

To add a new sign to the model's vocabulary, follow these three steps in sequence:

1.  Capture Images: Run `collectdata.py`. When the webcam opens, press the first letter of the new sign's name on the keyboard to save raw images into the Image/ folder. Press 'q' to stop.
2.  Extract Keypoints: Run `data.py` to convert those raw images into numeric keypoints, saving them into the **MP\_Data/** folder.
3.  Train Model: Run `trainmodel.py` to retrain the neural network with the old and new data, updating `model.h5`.







