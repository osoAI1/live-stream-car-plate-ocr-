from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

# Import the Sort class from the sort.py file you've uploaded
from sort import Sort
def read_license_plate(img):
    """Extracts text from a license plate image using EasyOCR."""
    results = reader.readtext(img)
    if results:
        best_result = max(results, key=lambda result: result[2])  # Find the result with the highest confidence
        return best_result[1], best_result[2]  # Return text and confidence score
    return None, None  # Ensure always returning a tuple

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load YOLO models
vehicle_detector = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Video stream setup
cap = cv2.VideoCapture(0)  # Change to '0' for webcam use or a video file path
mot_tracker = Sort()
results = []

# Define class IDs for vehicles (you may need to adjust these based on your model)
vehicles = [2, 3, 5, 7]  # Example class IDs for vehicles

# Process video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles
    vehicle_detections = vehicle_detector(frame)[0]
    vehicles_in_frame = [d for d in vehicle_detections.boxes.data.tolist() if int(d[5]) in vehicles]

    # Track vehicles
    tracked_vehicles = mot_tracker.update(np.asarray(vehicles_in_frame))

    # Detect license plates
    license_detections = license_plate_detector(frame)[0]
    license_plates = license_detections.boxes.data.tolist()

    for license_plate in license_plates:
        x1, y1, x2, y2, _, _ = map(int, license_plate)
        license_plate_crop = frame[y1:y2, x1:x2]
        # OCR on the license plate
        text, confidence = read_license_plate(license_plate_crop)
        if text:  # Check if text is not None
            results.append({
                'bbox': (x1, y1, x2, y2),
                'text': text,
                'confidence': confidence
            })
            # Optionally draw results on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        else:
            print("No text found in the license plate")

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
