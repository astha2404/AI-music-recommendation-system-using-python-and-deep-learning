import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load FER-2013 dataset
data = pd.read_csv("fer2013.csv")

# Parse pixels and reshape
pixels = data["pixels"].tolist()
width, height = 48, 48
faces = np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape(width, height) for pixel in pixels])
faces = faces / 255.0  # Normalize
faces = faces.reshape(faces.shape[0], 48, 48, 1)

# One-hot encode labels
emotions = pd.get_dummies(data['emotion']).values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 classes in FER-2013

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)

# Save model
model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")
