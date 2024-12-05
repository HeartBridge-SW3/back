import os
import cv2
import numpy as np
import time
from flask import Flask, Response, render_template, jsonify
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text, draw_bounding_box, apply_offsets
from utils.preprocessor import preprocess_input

# TensorFlow 경고 로그 비활성화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Flask 앱 초기화
app = Flask(__name__)

# Emotion 모델 경로 및 설정
emotion_model_path = 'C:/Users/ehfk1/AppData/Local/Programs/Python/Python312/Emotion/models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
emotion_offsets = (20, 40)
frame_window = 10

# 모델 로드
if not os.path.exists(emotion_model_path):
    raise FileNotFoundError(f"Model file not found: {emotion_model_path}")

emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# 얼굴 인식 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 전역 변수
emotion_window = []
emotion_logs = []  # 감정 기록
emotion_tracker = {"emotion": None, "start_time": None}  # 감정을 추적

def generate_frames():
    """카메라 프레임을 읽고 감정을 인식하여 반환합니다."""
    global emotion_logs
    cap = cv2.VideoCapture(1)  # 기본 카메라 사용
    while True:
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

            # 감정 추적 로직
            current_time = time.time()
            if emotion_tracker["emotion"] == emotion_mode:
                if current_time - emotion_tracker["start_time"] >= 1:  # 1초 이상 지속
                    if not emotion_logs or emotion_logs[-1] != emotion_mode:
                        emotion_logs.append(emotion_mode)  # 로그에 추가
                        if len(emotion_logs) > 10:  # 기록 제한
                            emotion_logs.pop(0)
            else:
                emotion_tracker["emotion"] = emotion_mode
                emotion_tracker["start_time"] = current_time

            # 감정에 따른 박스 색상
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

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """웹 페이지 렌더링."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """카메라 피드 스트리밍."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    """현재 감정 로그를 반환."""
    return jsonify(emotion_logs)

if __name__ == '__main__':
    app.run(debug=True)