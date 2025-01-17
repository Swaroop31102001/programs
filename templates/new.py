import cv2
import threading
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

# Email Configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_SENDER = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_email_password'
EMAIL_RECEIVER = 'receiver_email@gmail.com'

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 pre-trained model
cap = cv2.VideoCapture(0)  # Webcam for video capture

# Shared data between threads
people_count = 0
phone_detection_count = 0
lock = threading.Lock()
screenshot_paths = []


# Function to send email alerts
def send_email_alert(people_count, phone_detection_count, screenshot_paths):
    """
    Sends an email alert about people and phone detection with attached screenshots.

    Args:
        people_count (int): Number of people detected.
        phone_detection_count (int): Number of phones detected.
        screenshot_paths (list): List of file paths to the screenshots.
    """
    subject = "Proctoring Alert: People and Phone Detection Summary"
    body = f"""
    Dear Administrator,

    Here is the latest summary of the monitoring system:
    - Total People Detected: {people_count}
    - Total Phone Detections: {phone_detection_count}
    - Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Please review the attached screenshots for more details.

    Regards,
    Proctoring System
    """
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))

    # Attach each screenshot
    for screenshot_path in screenshot_paths:
        try:
            with open(screenshot_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(screenshot_path)}"')
                msg.attach(part)
        except Exception as e:
            print(f"Error attaching file {screenshot_path}: {e}")

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Alert email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")


# Function to detect people and phones
def detect_people_and_phones():
    global people_count, phone_detection_count, screenshot_paths

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Run YOLOv5 model
        results = model(frame)
        detections = results.pandas().xyxy[0]  # Detected objects

        # Counters
        detected_people = 0
        phone_detected = False

        # Iterate over detections
        for _, row in detections.iterrows():
            label = row['name']
            confidence = row['confidence']

            if label == 'person' and confidence > 0.3:
                detected_people += 1
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if label == 'cell phone' and confidence > 0.3:
                phone_detected = True
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Phone Detected ({confidence:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Update shared counters
        with lock:
            people_count += detected_people
            if phone_detected:
                phone_detection_count += 1
                screenshot_path = f"screenshot_{phone_detection_count}.jpg"
                cv2.imwrite(screenshot_path, frame)  # Save screenshot
                screenshot_paths.append(screenshot_path)
                print(f"Phone detected. Total count: {phone_detection_count}")

        # Display frame (optional for debugging)
        cv2.imshow("People and Phone Detection", frame)

        # Check if email needs to be sent
        with lock:
            if phone_detection_count > 3 or len(screenshot_paths) >= 3:
                send_email_alert(people_count, phone_detection_count, screenshot_paths)
                phone_detection_count = 0  # Reset counters
                screenshot_paths = []  # Clear screenshots

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Start detection in a separate thread
threading.Thread(target=detect_people_and_phones, daemon=True).start()

# Flask App (Optional: If you want to serve results via API)
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/people_count')
def get_people_count():
    with lock:
        return jsonify({'people_count': people_count})

@app.route('/phone_count')
def get_phone_count():
    with lock:
        return jsonify({'phone_count': phone_detection_count})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
