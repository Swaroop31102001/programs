import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device='cpu')  # Use YOLOv5x for better accuracy
model.conf = 0.5
model.iou = 0.5

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50, n_init=3, nn_budget=100)

def process_live_video():
    cap = cv2.VideoCapture(0)  # 0 for webcam, or you can replace with a video file path
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_skip = 10  # Process every 10th frame to reduce processing load
    unique_ids = set()
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize frame for consistency
        resize_width, resize_height = 1280, 720
        frame = cv2.resize(frame, (resize_width, resize_height))

        # Perform detection with YOLOv5
        results = model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # Class 0 = person
                x1, y1, x2, y2 = map(int, xyxy)
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to x, y, w, h
                detections.append((bbox, float(conf), 'Person'))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed():
                unique_ids.add(track.track_id)
                x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track.track_id}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the live frame with detections
        cv2.putText(frame, f'Unique persons: {len(unique_ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Live People Tracking', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script to start live tracking
if __name__ == "__main__":
    process_live_video()
