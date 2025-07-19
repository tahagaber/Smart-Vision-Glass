import cv2
import numpy as np
import torch
import logging
from typing import Optional
from config import Config

logger = logging.getLogger(__name__)

class ResourceManager:
    def __init__(self, config: Config):
        self.config = config
        self.face_recognizer = None
        self.face_cascade = None
        self.model = None
        self.label_map = None
        self.device = None
        
    def initialize(self) -> bool:
        try:
            # Initialize face recognizer
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.read(self.config.face_recognizer_path)
            
            # Load label map
            self.label_map = np.load(self.config.label_map_path, allow_pickle=True).item()
            
            # Initialize face cascade
            if Path(self.config.face_cascade_path).is_file():
                self.face_cascade = cv2.CascadeClassifier(self.config.face_cascade_path)
            else:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + self.config.face_cascade_path)
                
            # Initialize YOLOv5
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.model.to(self.device).eval()
            
            logger.info("Models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing resources: {str(e)}")
            return False

    def cleanup(self):
        try:
            if self.model:
                del self.model
            torch.cuda.empty_cache()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
