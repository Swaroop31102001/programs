import cv2
import numpy as np
from sort import Sort  # Assuming you've downloaded sort.py

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize SORT tracker
tracker = Sort()

# Load class names (for COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (0 for webcam or path to video file)
cap = cv2.VideoCapture(0)  # Change 0 to path of video if using a file

# Person class index in COCO dataset (person is class ID 0)
person_class_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Prepare the frame for YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Parse detections and collect bounding boxes for people
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == person_class_index:
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maxima suppression to avoid multiple bounding boxes for the same object
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        boxes = [boxes[i[0]] for i in indices]
    
    # Update SORT tracker with bounding boxes
    trackers = tracker.update(np.array(boxes))
    
    # Draw bounding boxes and update the count
    count = 0
    for tracker in trackers:
        x, y, w, h, track_id = tracker
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count += 1
    
    # Display the number of tracked people
    cv2.putText(frame, f'People Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow("Live People Tracking", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
