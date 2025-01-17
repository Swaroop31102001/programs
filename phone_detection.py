import torch
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import os

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_SENDER = 'swa31102001@gmail.com'
EMAIL_PASSWORD = 'wtiq ghhb txsm zbze'  
EMAIL_RECEIVER = 'swaroop102001@gmail.com' 

def send_email_alert(detection_count, screenshot_paths):
    """
    Sends an email alert about phone detection with attached screenshots.

    Args:
        detection_count (int): Number of times the phone was detected.
        screenshot_paths (list): List of file paths to the screenshots.
    """
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

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

phone_detection_count = 0
screenshot_count = 0
screenshot_paths = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    results = model(frame)
    detections = results.pandas().xyxy[0] 

    phone_detected = False

    # Iterate over detections
    for _, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']
        if label == 'cell phone' and confidence > 0.3:
            phone_detected = True
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0), 2)
            cv2.putText(frame, f"Phone Detected ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)

    if phone_detected:
        phone_detection_count += 1
        screenshot_count += 1
        screenshot_path = f"screenshot_{screenshot_count}.jpg"
        cv2.imwrite(screenshot_path, frame)  # Save the screenshot
        screenshot_paths.append(screenshot_path)
        print(f"Phone detected. Count: {phone_detection_count}")
        cv2.putText(frame, f"WARNING: Phone Usage Detected! Count: {phone_detection_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if phone_detection_count > 3:
        send_email_alert(phone_detection_count, screenshot_paths)
        phone_detection_count = 0  # Reset count after alert
        screenshot_paths = []  # Clear the list of screenshots

    cv2.imshow("Phone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for path in screenshot_paths:
    if os.path.exists(path):
        os.remove(path)
