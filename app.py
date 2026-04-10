import cv2
import mediapipe as mp
import numpy as np
import base64
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=5 * 1024 * 1024)  # 5 MB for frame data

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose       = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)

# ── Exercise config ────────────────────────────────────────────────────────────
EXERCISE_CONFIG = {
    "bicep_curl": {
        "landmarks"    : (11, 13, 15),   # Left: Shoulder, Elbow, Wrist
        "down_angle"   : 160,
        "up_angle"     : 30,
        "display_name" : "Bicep Curl",
        "video"        : "static/bicep_curl.mp4",
    },
    "squat": {
        "landmarks"    : (23, 25, 27),   # Left: Hip, Knee, Ankle
        "down_angle"   : 160,
        "up_angle"     : 90,
        "display_name" : "Squat",
        "video"        : "static/squat.mp4",
    },
    "pushup": {
        "landmarks"    : (11, 13, 15),   # Left: Shoulder, Elbow, Wrist
        "down_angle"   : 150,
        "up_angle"     : 80,
        "display_name" : "Push-Up",
        "video"        : "static/pushup.mp4",
    },
}

# ── Per-client state ───────────────────────────────────────────────────────────
# Keyed by socket session id (sid)
client_state = {}

def get_state(sid):
    if sid not in client_state:
        client_state[sid] = {
            "counter"          : 0,
            "stage"            : None,
            "current_exercise" : "bicep_curl",
        }
    return client_state[sid]


# ── Helpers ───────────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c  = np.array(a), np.array(b), np.array(c)
    radians  = (np.arctan2(c[1] - b[1], c[0] - b[0])
                - np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle    = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def landmarks_to_list(pose_landmarks):
    if pose_landmarks is None:
        return []
    return [
        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
        for lm in pose_landmarks.landmark
    ]


# ── Socket events ─────────────────────────────────────────────────────────────
from flask_socketio import disconnect
from flask import request

@socketio.on('connect')
def on_connect():
    sid = request.sid
    get_state(sid)  # init state
    print(f"✅ Client connected: {sid}")


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    client_state.pop(sid, None)
    print(f"🔌 Client disconnected: {sid}")


@socketio.on('set_exercise')
def change_exercise(data):
    sid   = request.sid
    state = get_state(sid)
    name  = data.get('name', 'bicep_curl')
    if name not in EXERCISE_CONFIG:
        return

    state["current_exercise"] = name
    state["counter"]          = 0
    state["stage"]            = None

    print(f"▶ [{sid[:6]}] Exercise → {name}")
    emit('exercise_changed', {
        'exercise'    : name,
        'display_name': EXERCISE_CONFIG[name]['display_name'],
    })


@socketio.on('frame')
def process_frame(data):
    """
    Receive a base64 JPEG frame from the browser,
    run MediaPipe pose detection, and emit back landmarks + rep count.
    """
    sid   = request.sid
    state = get_state(sid)

    try:
        # Decode base64 → numpy image
        img_b64  = data.get('image', '')
        if not img_b64:
            return

        # Strip data URI prefix if present
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        img_bytes = base64.b64decode(img_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # NOTE: Browser already mirrors the video (CSS scaleX(-1)),
        # but sends the RAW (un-mirrored) frame. We flip here so MediaPipe
        # sees the same orientation the user sees on screen.
        frame = cv2.flip(frame, 1)

        image                 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results               = pose.process(image)
        image.flags.writeable = True

        exercise    = state["current_exercise"]
        config      = EXERCISE_CONFIG[exercise]
        i1, i2, i3  = config["landmarks"]
        down_thresh = config["down_angle"]
        up_thresh   = config["up_angle"]

        angle     = 0.0
        landmarks = []

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            a = [lm[i1].x, lm[i1].y]
            b = [lm[i2].x, lm[i2].y]
            c = [lm[i3].x, lm[i3].y]

            angle     = calculate_angle(a, b, c)
            landmarks = landmarks_to_list(results.pose_landmarks)

            if angle > down_thresh:
                state["stage"] = "down"
            if angle < up_thresh and state["stage"] == "down":
                state["stage"]   = "up"
                state["counter"] += 1
                print(f"💪 [{sid[:6]}] REP #{state['counter']}  ({exercise})")

        emit('data_update', {
            'reps'     : state["counter"],
            'angle'    : round(angle, 1),
            'stage'    : state["stage"],
            'exercise' : exercise,
            'landmarks': landmarks,
            'detected' : bool(results.pose_landmarks),
        })

    except Exception as e:
        print(f"Frame error [{sid[:6]}]: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debug')
def debug():
    return """
    <!DOCTYPE html><html><head><title>Debug</title>
    <style>
      body{background:#111;color:#0f0;font-family:monospace;padding:20px}
      #status{font-size:20px;font-weight:bold;margin:8px 0}
      #info{font-size:14px;color:#aaa;margin:6px 0}
      .ok{color:#0f0} .fail{color:#f44}
    </style></head><body>
    <h2>🔍 MediaPipe Debug (Browser-frame mode)</h2>
    <div id="status" class="fail">Connecting…</div>
    <div id="info">—</div>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
      const s = io();
      s.on('data_update', d => {
        const el = document.getElementById('status');
        el.textContent = d.detected ? '✅ POSE DETECTED' : '❌ No pose — check lighting / framing';
        el.className   = d.detected ? 'ok' : 'fail';
        document.getElementById('info').textContent =
          'Exercise: ' + d.exercise +
          ' | Angle: ' + d.angle + '°' +
          ' | Stage: ' + d.stage +
          ' | Reps: ' + d.reps;
      });
    </script>
    </body></html>
    """


if __name__ == '__main__':
    socketio.run(app, port=5000, debug=False)