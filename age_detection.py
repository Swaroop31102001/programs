import cv2
import numpy as np

# Load the pre-trained models
age_net = cv2.dnn.readNetFromCaffe(
    r'C:\Users\swaro\.vscode\yolov5-master\age_deploy.prototxt',
    r'C:\Users\swaro\.vscode\yolov5-master\RFB-320.caffemodel'
)

# Age categories
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Function to detect faces and estimate age
def detect_and_estimate_age(frame):
    h, w = frame.shape[:2]

    # Convert frame to blob for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    age_net.setInput(blob)

    # Forward pass for detection
    detections = age_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            # Get coordinates for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure coordinates are within the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:  # Skip if the ROI is invalid
                continue

            # Create blob for age estimation
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), 
                (78.4263377603, 87.7689143744, 114.895847746), 
                swapRB=False, crop=False
            )
            age_net.setInput(face_blob)

            # Predict age
            age_preds = age_net.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]
            confidence_score = age_preds[0].max()

            # Draw bounding box and label
            label = f"Age: {age}, Confidence: {confidence_score:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

print("[INFO] Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame. Exiting...")
        break

    # Detect and estimate age
    output_frame = detect_and_estimate_age(frame)

    # Display the output
    cv2.imshow("Real-Time Age Detection", output_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting video stream.")
        break

cap.release()
cv2.destroyAllWindows()
