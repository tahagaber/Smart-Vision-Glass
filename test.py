import cv2
import numpy as np
import os
import time
import face_recognition
from threading import Thread
import tkinter as tk
from ultralytics import YOLO
import torch
import pyttsx3

# === إعداد المحرك الصوتي ===
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# === تحميل نموذج YOLOv8 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov8n.pt')
print("✅ YOLOv8 model loaded successfully")

# === تحميل الصور من مجلد الوجوه ===
known_face_encodings = []
known_face_names = []
face_images_path = r"D:\data set\Taha_Gaber"  # ← عدل المسار حسب مكان الصور

def load_known_faces():
    for filename in os.listdir(face_images_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(face_images_path, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"✅ Loaded: {name}")
            else:
                print(f"❌ No face found in {filename}")

# === تعريف ألوان HSV ===
color_ranges = {
    'red': [(0, 150, 100), (10, 255, 255), (170, 150, 100), (180, 255, 255)],
    'green': [(35, 100, 100), (85, 255, 255)],
    'blue': [(90, 100, 100), (130, 255, 255)],
    'yellow': [(20, 100, 100), (35, 255, 255)],
    'black': [(0, 0, 0), (180, 255, 50)],
    'white': [(0, 0, 150), (180, 30, 255)],
}
color_bgr = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

# === الوضع الحالي ===
mode = "none"
running = True

def set_mode(new_mode):
    global mode
    mode = new_mode
    print(f"🔄 Switched to mode: {mode}")

    spoken_texts = {
        "face": "تم تفعيل التعرف على الوجه",
        "object": "تم تفعيل التعرف على الكائنات",
        "color": "تم تفعيل كشف اللون",
        "all": "تم تفعيل جميع الوظائف",
        "none": "تم إيقاف الوظائف مؤقتًا"
    }
    if new_mode in spoken_texts:
        speak(spoken_texts[new_mode])

def stop_program():
    global running
    running = False
    speak("تم إنهاء البرنامج")
    print("🛑 Program stopped")

# === الكاميرا ومعالجة الفيديو ===
def video_loop():
    load_known_faces()
    cap = cv2.VideoCapture(0)
    time.sleep(1)

    last_sentence = ""

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        detected_name = ""
        detected_objects = []
        detected_color = ""

        # === التعرف على الوجوه ===
        if mode in ["face", "all"]:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            for (top, right, bottom, left), face_encoding in zip(locations, encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    name = known_face_names[best_match_index]
                    detected_name = name

                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # === التعرف على الكائنات ===
        if mode in ["object", "all"]:
            results = model.predict(frame, conf=0.5, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results.names[int(box.cls[0])]
                detected_objects.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # === كشف اللون ===
        if mode in ["color", "all"]:
            h, w = frame.shape[:2]
            center = frame[h//2-50:h//2+50, w//2-50:w//2+50]
            hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            detected = {}
            for name, ranges in color_ranges.items():
                mask = None
                for i in range(0, len(ranges), 2):
                    lower = np.array(ranges[i])
                    upper = np.array(ranges[i+1])
                    temp_mask = cv2.inRange(hsv, lower, upper)
                    mask = temp_mask if mask is None else cv2.bitwise_or(mask, temp_mask)
                if cv2.countNonZero(mask) > 300:
                    detected[name] = cv2.countNonZero(mask)

            if detected:
                top_color = max(detected, key=detected.get)
                detected_color = top_color
                cv2.rectangle(frame, (w//2-50, h//2-50), (w//2+50, h//2+50), color_bgr[top_color], 2)
                cv2.putText(frame, f"Color: {top_color}", (w//2-60, h//2+70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr[top_color], 2)

        # === تكوين الجملة المنطوقة ===
        sentence_parts = []

        if detected_name:
            clean_name = detected_name.split()[0].split("_")[0].split("(")[0]
            sentence_parts.append(f"الشخص اللي قدامك هو {clean_name}")

        if detected_objects:
            objs = " و ".join(set(detected_objects))
            sentence_parts.append(f"ويوجد {objs}")

        if detected_color:
            sentence_parts.append(f"واللون {detected_color}")

        final_sentence = ". ".join(sentence_parts)

        if final_sentence and final_sentence != last_sentence:
            speak(final_sentence)
            last_sentence = final_sentence

        # عرض الكاميرا
        cv2.imshow("📸 AI Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# === واجهة التحكم ===
root = tk.Tk()
root.title("🔧 AI Vision Controller")
tk.Label(root, text="اختر الوظيفة:", font=("Arial", 14)).pack(pady=10)
tk.Button(root, text="👤 التعرف على الوجه", command=lambda: set_mode("face"), width=30).pack(pady=5)
tk.Button(root, text="📦 التعرف على الكائنات", command=lambda: set_mode("object"), width=30).pack(pady=5)
tk.Button(root, text="🎨 كشف اللون", command=lambda: set_mode("color"), width=30).pack(pady=5)
tk.Button(root, text="🌀 الكل معًا", command=lambda: set_mode("all"), width=30).pack(pady=5)
tk.Button(root, text="⏸️ إيقاف مؤقت", command=lambda: set_mode("none"), width=30).pack(pady=5)
tk.Button(root, text="❌ خروج", command=stop_program, width=30).pack(pady=10)

Thread(target=video_loop, daemon=True).start()
root.mainloop()
