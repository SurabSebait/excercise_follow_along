import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Exercise configurations: landmark indices + angle thresholds
EXERCISE_CONFIG = {
    "bicep_curl": {
        "landmarks": (11, 13, 15),  # Left: Shoulder, Elbow, Wrist
        "down_angle": 160,
        "up_angle": 30,
        "display_name": "Bicep Curl"
    },
    "squat": {
        "landmarks": (23, 25, 27),  # Left: Hip, Knee, Ankle
        "down_angle": 170,
        "up_angle": 90,
        "display_name": "Squat"
    },
    "pushup": {
        "landmarks": (11, 13, 15),  # Left: Shoulder, Elbow, Wrist
        "down_angle": 160,
        "up_angle": 70,
        "display_name": "Push-Up"
    }
}

# Global State
state = {
    "counter": 0,
    "stage": None,
    "current_exercise": "bicep_curl"
}


def calculate_angle(a, b, c):
    """Calculate the angle at point b formed by a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


@socketio.on('set_exercise')
def change_exercise(data):
    name = data.get('name', 'bicep_curl')
    if name not in EXERCISE_CONFIG:
        print(f"Unknown exercise: {name}. Ignoring.")
        return
    state["current_exercise"] = name
    state["counter"] = 0
    state["stage"] = None
    print(f"Exercise switched to: {state['current_exercise']}")
    emit('exercise_changed', {
        'exercise': name,
        'display_name': EXERCISE_CONFIG[name]['display_name']
    })


def background_process():
    cap = cv2.VideoCapture("static/pushup.mp4")

    if not cap.isOpened():
        print("ERROR: Could not open video file: static/pushup.mp4")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                exercise = state["current_exercise"]

                # Guard: skip if exercise config not found
                if exercise not in EXERCISE_CONFIG:
                    socketio.sleep(0.03)
                    continue

                config = EXERCISE_CONFIG[exercise]
                i1, i2, i3 = config["landmarks"]
                down_thresh = config["down_angle"]
                up_thresh = config["up_angle"]

                # Extract landmark coordinates
                a = [lm[i1].x, lm[i1].y]
                b = [lm[i2].x, lm[i2].y]
                c = [lm[i3].x, lm[i3].y]

                angle = calculate_angle(a, b, c)

                # Rep counter logic
                if angle > down_thresh:
                    state["stage"] = "down"
                if angle < up_thresh and state["stage"] == "down":
                    state["stage"] = "up"
                    state["counter"] += 1

                socketio.emit('data_update', {
                    'reps': state["counter"],
                    'angle': round(angle, 1),
                    'stage': state["stage"],
                    'exercise': exercise
                })

        except Exception as e:
            print(f"Frame processing error: {e}")

        socketio.sleep(0.03)

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.start_background_task(background_process)
    socketio.run(app, port=5000, debug=False)