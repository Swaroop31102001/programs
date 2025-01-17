import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, render_template, request, jsonify
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import time

# Email Configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_SENDER = 'swa31102001@gmail.com'
EMAIL_PASSWORD = 'wtiq ghhb txsm zbze'  
EMAIL_RECEIVER = 'swaroop102001@gmail.com'

app = Flask(__name__)

# Initialize YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device='cpu')  # YOLOv5x for people and phone detection
model.conf = 0.5
model.iou = 0.5

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=50, n_init=3, nn_budget=100)

# Shared Variables
stop_event = threading.Event()
detection_results = {
    "unique_people_count": 0,
    "phone_detections": 0
}

def send_email_report(people_count, phone_count):
    """Send an email with the counts of people and phone detections."""
    subject = "Detection Report: People and Phone Usage"
    body = f"""
    Dear Admin,

    Here is the detection report:
    - Unique People Count: {people_count}
    - Phone Detections: {phone_count}

    Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
    """
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

def detect_people_and_phones():
    """Detect and count unique people and phone detections."""
    cap = cv2.VideoCapture(0)  # Webcam capture
    unique_ids = set()
    phone_detection_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # YOLOv5 detection
        results = model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            cls_id = int(cls)
            x1, y1, x2, y2 = map(int, xyxy)

            if cls_id == 0:  # Class 0 = person
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox, float(conf), 'Person'))
            elif cls_id == 67:  # Class 67 = cell phone
                phone_detection_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Phone Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Update tracker for people
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed():
                unique_ids.add(track.track_id)
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display frame
        cv2.putText(frame, f"People Count: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Phone Detections: {phone_detection_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Live Tracking", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    detection_results["unique_people_count"] = len(unique_ids)
    detection_results["phone_detections"] = phone_detection_count

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    """Start the detection process."""
    stop_event.clear()
    detection_thread = threading.Thread(target=detect_people_and_phones)
    detection_thread.start()
    return render_template('exam.html')

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    answers = request.form.to_dict()
    stop_event.set()
    send_email_report(
        detection_results["unique_people_count"],
        detection_results["phone_detections"]
    )
    return render_template('thank.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)