import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import os
import random

# Load the model without the optimizer (compile=False)
def load_model_without_optimizer(model_path):
    model = load_model(model_path, compile=False)
    return model

# Path to your saved model
model_path = r'C:\Users\singh\OneDrive\Desktop\music recommender 2\music recommender\model.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    print("Model file not found at:", model_path)
else:
    print("Model found, loading...")

# Load the model without optimizer
model = load_model_without_optimizer(model_path)

# Emotion labels (ensure these match your model's output classes)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Path to the music folder
music_dir = r'C:\Users\singh\OneDrive\Desktop\music recommender 2\music recommender\music'

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load the face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize pygame mixer for playing music
pygame.mixer.init()

def get_random_song(emotion_folder):
    """Select a random song from the given emotion folder."""
    emotion_folder_path = os.path.join(music_dir, emotion_folder)
    if os.path.exists(emotion_folder_path):
        songs = [f for f in os.listdir(emotion_folder_path) if f.endswith('.mp3')]
        if songs:
            return os.path.join(emotion_folder_path, random.choice(songs))
        else:
            print(f"No songs found for emotion: {emotion_folder}")
            return None
    else:
        print(f"No folder found for emotion: {emotion_folder}")
        return None

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the frame
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.astype('float32') / 255
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Predict the emotion
        prediction = model.predict(face_roi)
        emotion = emotion_labels[np.argmax(prediction)]

        # Display emotion on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Stop webcam and play music when face is detected
        print("Face detected, stopping webcam...")
        cap.release()  # Release the webcam

        # Get the random song for the detected emotion
        music_file = get_random_song(emotion)

        if music_file:
            print(f"Playing {music_file} for emotion: {emotion}")
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play()
        else:
            print(f"Error: No music file found for emotion: {emotion}")

        # Show the result
        cv2.imshow("Emotion Detection", frame)

        # Wait until the window is closed manually
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Exit the program cleanly
        exit()  # or use `break` if needed outside the loop

# Cleanup (only reached if loop exits without emotion detected)
cap.release()
cv2.destroyAllWindows()


