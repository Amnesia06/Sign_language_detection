from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import os
import numpy as np

# List of phrases for which hand landmarks data is collected
phrases = ["Thankyou", "Welcome", "Hello", "Bye", "Peace", "Yes", "No", "ILoveYou", "Care"]

label_map = {label: num for num, label in enumerate(phrases)}

DATA_PATH = 'MP_Data'  # Path where hand landmarks data is saved

frames, labels = [], []

# Load hand landmarks data and corresponding labels
for phrase in phrases:
    phrase_path = os.path.join(DATA_PATH, phrase)
    for frame_num in range(len(os.listdir(phrase_path))):
        res = np.load(os.path.join(phrase_path, f'{frame_num}.npy'))
        frames.append(res)
        labels.append(label_map[phrase])

X = np.array(frames)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the LSTM model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(63,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(phrases), activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Save the trained model weights
model.save('model.h5')

# Save the model architecture to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
