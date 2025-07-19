import cv2
import numpy as np
import time
import torch
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_vision.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        # Initialize models
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read(r'D:\project\pythonProject1\faces_recognizer999.xml')
        self.label_map = np.load(r'D:\project\pythonProject1\label_map999.npy', allow_pickle=True).item()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize YOLOv5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.model.to(self.device).eval()
        
        # Initialize state
        self.mode = "none"
        self.running = True
        self.last_detection_time = time.time()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Color ranges
        self.color_ranges = {
            'red': [(0, 150, 50), (10, 255, 255), (170, 150, 50), (180, 255, 255)],
            'green': [(35, 80, 50), (85, 255, 255)],
            'blue': [(90, 100, 50), (130, 255, 255)],
            'yellow': [(20, 100, 50), (35, 255, 255)],
        }
        self.color_bgr = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
        }

    def process_frame(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            if self.mode in ["face", "all"]:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h) in faces:
                    roi = gray[y:y + h, x:x + w]
                    try:
                        resized = cv2.resize(roi, (150, 150))
                        label, confidence = self.face_recognizer.predict(resized)
                        name = self.label_map.get(label, "Unknown") if confidence < 75 else "Unknown"
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{name} ({int(confidence)}%)", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except Exception as e:
                        logger.error(f"Face recognition error: {str(e)}")
                        continue

            # Color detection
            if self.mode in ["color", "all"]:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, w = frame.shape[:2]
                roi = hsv[h // 3:h * 2 // 3, w // 3:w * 2 // 3]
                dominant = "Unknown"
                max_pixels = 0

                for cname, ranges in self.color_ranges.items():
                    mask = None
                    for i in range(0, len(ranges), 2):
                        lower = np.array(ranges[i])
                        upper = np.array(ranges[i + 1])
                        part = cv2.inRange(roi, lower, upper)
                        mask = part if mask is None else cv2.bitwise_or(mask, part)
                    pixels = cv2.countNonZero(mask)
                    if pixels > max_pixels:
                        max_pixels = pixels
                        dominant = cname

                if dominant != "Unknown":
                    cv2.putText(frame, f"Color: {dominant}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_bgr[dominant], 2)

            # Object detection
            current_time = time.time()
            if self.mode in ["object", "all"] and (current_time - self.last_detection_time > 1.0):
                try:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.model(img_rgb, size=416)

                    for *xyxy, conf, cls in results.xyxy[0].cpu():
                        name = self.model.names[int(cls)]
                        x1, y1, x2, y2 = map(int, xyxy)
                        confidence = float(conf)
                        if confidence > 0.5:
                            label = f"{name} {confidence:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.6, (255, 200, 0), 2)

                    self.last_detection_time = current_time
                except Exception as e:
                    logger.error(f"Object detection error: {str(e)}")

            # FPS display
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1:
                fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            return frame

        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return frame

    def run(self):
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 24)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow("📷 AI Vision Live", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera stopped")

        except Exception as e:
            logger.error(f"Main loop error: {str(e)}")
            if cap:
                cap.release()
            cv2.destroyAllWindows()

    def set_mode(self, new_mode):
        self.mode = new_mode
        logger.info(f"Mode changed to: {new_mode}")

    def stop(self):
        self.running = False
        logger.info("Program stopping...")
