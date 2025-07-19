import cv2
import os
import numpy as np
from tqdm import tqdm


class FaceRecognitionSystem:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.image_size = (300, 300)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        """Detect faces in an image using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(gray[y:y + h, x:x + w], (x, y, w, h)) for (x, y, w, h) in faces]

    def process_dataset(self):
        """Process the dataset and train the recognizer"""
        faces = []
        labels = []
        person_names = os.listdir(self.data_path)

        print("\nProcessing dataset...")
        for label, person_name in enumerate(person_names):
            person_folder = os.path.join(self.data_path, person_name)
            images = os.listdir(person_folder)

            print(f"\nProcessing {person_name} ({len(images)} images)")
            for image_name in tqdm(images, desc=f"Images for {person_name}"):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Warning: Could not load image: {image_path}")
                    continue

                detected_faces = self.detect_faces(image)
                if not detected_faces:
                    print(f"Warning: No face detected in: {image_path}")
                    continue

                for (face, _) in detected_faces:
                    try:
                        resized_face = cv2.resize(face, self.image_size)
                        faces.append(resized_face)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing face in {image_path}: {str(e)}")

        return np.array(faces), np.array(labels)

    def train_model(self):
        """Train and save the face recognition model"""
        print("\nTraining model...")
        faces, labels = self.process_dataset()

        if len(faces) == 0:
            print("No faces found in the dataset!")
            return False

        face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )

        face_recognizer.train(faces, labels)

        # Save the model
        model_save_path = os.path.join(self.output_dir, 'faces_recognizer_taha_15.xml')
        face_recognizer.save(model_save_path)

        # Save the label map
        label_map = {label: person_name for label, person_name in enumerate(os.listdir(self.data_path))}
        label_map_save_path = os.path.join(self.output_dir, 'label_map_taha_15.npy')
        np.save(label_map_save_path, label_map)

        print("\nModel training completed successfully!")
        return True


if __name__ == "__main__":
    # Initialize the face recognition system
    data_path = r"D:\data set\Taha"
    output_dir = r"E:\New folder"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create and train the system
    face_system = FaceRecognitionSystem(data_path, output_dir)
    success = face_system.train_model()

    if success:
        print("\nModel and label map saved successfully!")
    else:
        print("\nFailed to train the model.")
