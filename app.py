from flask import Flask, render_template, request, redirect, url_for, session, Response
import sqlite3
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import threading
from playsound import playsound
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'

DB_NAME = 'database.db'

# Initialize DB
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        action TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

# Global variables
running = False
alarm_status = False
alarm_status2 = False
COUNTER = 0
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20

detector = cv2.CascadeClassifier(r"C:\Users\patha\OneDrive\Desktop\Real-Time-Drowsiness-Detection-System\detection\haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(r"C:\Users\patha\OneDrive\Desktop\Real-Time-Drowsiness-Detection-System\shape_predictor_68_face_landmarks.dat")

cap = None
video_thread = None

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    return (leftEAR + rightEAR) / 2.0

def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

def sound_alarm(path):
    global alarm_status, alarm_status2
    while alarm_status:
        playsound(path)
    if alarm_status2:
        playsound(path)

def generate_frames():
    global COUNTER, alarm_status, alarm_status2
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (450, int(frame.shape[0] * 450 / frame.shape[1])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            ear = final_ear(shape)
            distance = lip_distance(shape)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        threading.Thread(target=sound_alarm, args=(r"C:\Users\patha\OneDrive\Desktop\Real-Time-Drowsiness-Detection-System\static\sounds\Alert.wav",), daemon=True).start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                if not alarm_status2:
                    alarm_status2 = True
                    threading.Thread(target=sound_alarm, args=("static/sounds/Alert.wav",), daemon=True).start()
                cv2.putText(frame, "Yawn Alert", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                alarm_status2 = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = 'Username already exists.'
    return render_template('signup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
            user = cursor.fetchone()
            if user:
                session['username'] = username
                return redirect(url_for('dashboard'))
            else:
                error = 'Invalid Credentials'
    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Add the 'detection_running' status based on the global `running` variable
    return render_template('dashboard.html', username=session['username'], detection_running=running)


@app.route('/start_detection', methods=['POST'])
def start():
    global running, cap
    if not running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Could not access the webcam"
        running = True
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute('INSERT INTO sessions (username, action) VALUES (?, ?)', (session.get('username', 'anonymous'), 'start'))
            conn.commit()

    return redirect(url_for('dashboard'))

@app.route('/stop_detection', methods=['POST'])
def stop():
    global running, cap
    if running:
        running = False
        if cap and cap.isOpened():
            cap.release()
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute('INSERT INTO sessions (username, action) VALUES (?, ?)', (session.get('username', 'anonymous'), 'stop'))
            conn.commit()

    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
