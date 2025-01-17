from flask import Flask, render_template, request, redirect, url_for, jsonify
import os, torch, cv2, threading, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import time

app = Flask(__name__)

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_SENDER = 'swa31102001@gmail.com'
EMAIL_PASSWORD = 'wtiq ghhb txsm zbze'  
EMAIL_RECEIVER = 'swaroop102001@gmail.com'

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
dummy_frame = torch.zeros((1, 3, 640, 480))
model(dummy_frame)

stop_event = threading.Event()
proctoring_active = threading.Event()

def send_email_alert(detection_count, screenshot_paths):
    subject = "Proctoring Alert: Phone Usage Detected"
    body = f"""
    Dear Candidate,

    You have been detected using your phone during the online exam.
    Detection Count: {detection_count}
    Last Detected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Please adhere to the examination guidelines.

    Regards,
    Proctoring System
    """
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    for screenshot_path in screenshot_paths:
        with open(screenshot_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(screenshot_path)}"')
            msg.attach(part)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Error sending email: {e}")

def phone_detection(stop_event, proctoring_active):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    cap = cv2.VideoCapture(0)

    phone_detection_count = 0
    screenshot_count = 0
    screenshot_paths = []

    proctoring_active.set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame)
        detections = results.pandas().xyxy[0]

        phone_detected = False
        for _, row in detections.iterrows():
            label = row['name']
            confidence = row['confidence']
            if label == 'cell phone' and confidence > 0.3:
                phone_detected = True
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Phone Detected ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if phone_detected:
            phone_detection_count += 1
            screenshot_count += 1
            screenshot_path = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(screenshot_path, frame)
            screenshot_paths.append(screenshot_path)

        if phone_detection_count > 3:
            send_email_alert(phone_detection_count, screenshot_paths)
            phone_detection_count = 0
            screenshot_paths = []

        if stop_event.is_set() or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    proctoring_active.clear()
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    stop_event.clear()
    detection_thread = threading.Thread(target=phone_detection, args=(stop_event, proctoring_active))
    detection_thread.daemon = True
    detection_thread.start()
    time.sleep(3)
    return redirect(url_for('check_proctoring_status'))

@app.route('/check_proctoring_status')
def check_proctoring_status():
    time.sleep(2)
    if proctoring_active.is_set():
        return redirect(url_for('exam_page'))
    return jsonify({"status": "starting", "details": "Proctoring is initializing..."}), 200

@app.route('/exam')
def exam_page():
    return render_template('exam.html')

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    answers = request.form.to_dict()
    stop_event.set()
    
    return render_template('thank.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
