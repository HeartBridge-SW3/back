import cv2
import numpy as np
import time
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

USE_WEBCAM = True  # If false, loads video file source

# Parameters for loading data and images
emotion_model_path = 'C:/Users/ehfk1/AppData/Local/Programs/Python/Python312/Emotion/models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# Hyperparameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# Loading models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure the model path exists
if not os.path.exists(emotion_model_path):
    raise FileNotFoundError(f"Model file not found: {emotion_model_path}")

# Load the emotion classification model
emotion_classifier = load_model(emotion_model_path, compile=False)

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Starting lists for calculating modes
emotion_window = []
emotion_tracker = {"emotion": None, "start_time": None}  # To track persistent emotions
emotion_logs = []  # To store logs of emotion changes

# Starting video streaming
cv2.namedWindow('Camera Feed')
cv2.namedWindow('Emotion Logs')
cap = cv2.VideoCapture(1) if USE_WEBCAM else cv2.VideoCapture('./demo/dinner.mp4')

while cap.isOpened():
    ret, bgr_image = cap.read()
    if not ret:
        break

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except cv2.error:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # Update emotion tracker
        current_time = time.time()
        if emotion_tracker["emotion"] == emotion_mode:
            if current_time - emotion_tracker["start_time"] >= 1:
                # Add to logs if not already added
                if not emotion_logs or emotion_logs[-1] != emotion_mode:
                    emotion_logs.append(emotion_mode)
                    if len(emotion_logs) > 10:  # Limit logs to last 10 entries
                        emotion_logs.pop(0)
        else:
            emotion_tracker["emotion"] = emotion_mode
            emotion_tracker["start_time"] = current_time

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int).tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    # Convert camera feed to BGR format
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Create a separate black image for emotion logs
    log_image = np.zeros((400, 300, 3), dtype=np.uint8)
    y_offset = 20
    for log in emotion_logs[::-1]:  # Display latest logs at the top
        cv2.putText(log_image, f"emotion: {log}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 30

    # Show camera feed and logs in separate windows
    cv2.imshow('Camera Feed', bgr_image)
    cv2.imshow('Emotion Logs', log_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
