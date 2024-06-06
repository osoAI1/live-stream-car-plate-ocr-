import customtkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import easyocr
from sort import Sort

reader = easyocr.Reader(['en'], gpu=False)

def read_license_plate(img):
    """Extracts text from a license plate image using EasyOCR."""
    results = reader.readtext(img)
    if results:
        best_result = max(results, key=lambda result: result[2])  # Find the result with the highest confidence
        return best_result[1], best_result[2]  # Return text and confidence score
    return None, None  # Ensure always returning a tuple

def calculate_iou(boxA, boxB):
    # Calculate intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate intersection area
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate union area
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    union_area = boxA_area + boxB_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

class WebcamApp:
    def __init__(self, window, cap):
        self.window = window
        self.window.minsize(800,670)
        self.window.resizable(False, False)
        self.vid = cap
        self.canvas = tk.CTkCanvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                    height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.place(x=185,y=290)
        self.delay = 10
        self.results = []
        self.vehicles = [2, 3, 5, 7]
        self.mot_tracker = Sort()
        self.update()

    def update(self):
        ret, img = self.vid.read()
       
        if ret:
            vehicle_detections = vehicle_detector(img)[0]
            vehicles_in_frame = [d for d in vehicle_detections.boxes.data.tolist() if int(d[5]) in self.vehicles]
            tracked_vehicles = self.mot_tracker.update(np.asarray(vehicles_in_frame))
            license_detections = license_plate_detector(img)[0]
            license_plates = license_detections.boxes.data.tolist()
            for license_plate in license_plates:
                x1, y1, x2, y2, _, _ = map(int, license_plate)
                license_plate_crop = img[y1:y2, x1:x2]
                # OCR on the license plate
                text, confidence = read_license_plate(license_plate_crop)
                if text:  # Check if text is not None
                    self.results.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': text,
                        'confidence': confidence
                    })
                    # Calculate IoU with ground truth
                    ground_truth = [(x1, y1, x2, y2)]  # Assuming ground truth is available
                    ious = [calculate_iou(license_plate, gt_box) for gt_box in ground_truth]
                    average_iou = sum(ious) / len(ious)
                    print("Average IoU:", average_iou)
                    # Display average IoU on the frame
                    cv2.putText(img, f'Average IoU: {average_iou:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # Optionally draw results on the frame
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    print("No text found in the license plate")

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(self.delay, self.update)

root = tk.CTk()
root.config(bg='#0A1D56')
root.title('Computer visiton')

butBack = tk.CTkButton(root, text='Back', font=('Jomhuria',35), corner_radius=25, fg_color='#279EFF',
                    hover_color='#135D66', border_color='black', border_width=2)#279EFF
butBack.place(x=90,y=35)

cap = cv2.VideoCapture(0)  # Change to '0' for webcam use or a video file path

# Load YOLO models
vehicle_detector = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector (2).pt')

app = WebcamApp(root, cap)
root.mainloop()

# Release resources when done
cap.release()
