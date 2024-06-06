import customtkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import easyocr
from sort_module import Sort  # Importing from the newly created sort_module

class LicensePlateReaderApp:
    def __init__(self, root):
        self.root = root
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.setup_ui()

    def setup_ui(self):
        self.root.title("License Plate Reader")
        self.root.geometry("800x600")
        self.lang_combobox = tk.CTkComboBox(self.root, values=["en", "fr", "de"], command=self.combobox_callback)
        self.lang_combobox.set("en")
        self.lang_combobox.place(x=50, y=50)

    def combobox_callback(self, choice):
        self.reader = easyocr.Reader([choice], gpu=False)
        print("Language changed to:", choice)

    def read_license_plate(self, img):
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            img_thresh = cv2.medianBlur(img_thresh, 5)
            img_processed = cv2.bitwise_not(img_thresh)

            results = self.reader.readtext(img_processed)
            if results:
                best_result = max(results, key=lambda result: result[2])
                text = best_result[1]
                confidence = best_result[2]
                self.display_text(text)
                print("Confidence Score:", confidence)
                return text, confidence
        except Exception as e:
            print("Error reading license plate:", e)
        return None, None

    def display_text(self, text):
        label = tk.CTkLabel(self.root, text=text, font=('Jomhuria', 40), text_color='#00cec9')
        label.place(x=685, y=410)

def main():
    root = tk.CTk()
    app = LicensePlateReaderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
