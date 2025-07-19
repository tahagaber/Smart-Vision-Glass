import tkinter as tk
from threading import Thread
from video_processor import VideoProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_vision.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 AI Vision Controller")
        self.root.geometry("300x400")
        
        # Initialize video processor
        self.processor = VideoProcessor()
        
        # Create UI elements
        self.create_widgets()
        
        # Start video processing in a separate thread
        self.start_video_thread()

    def create_widgets(self):
        # Mode selection buttons
        tk.Label(self.root, text="اختر وظيفة:", font=("Arial", 12)).pack(pady=10)
        
        self.face_btn = tk.Button(
            self.root, 
            text="👤 التعرف على الوجه",
            width=25,
            command=lambda: self.processor.set_mode("face")
        )
        self.face_btn.pack(pady=5)
        
        self.object_btn = tk.Button(
            self.root,
            text="📦 التعرف على الكائنات",
            width=25,
            command=lambda: self.processor.set_mode("object")
        )
        self.object_btn.pack(pady=5)
        
        self.color_btn = tk.Button(
            self.root,
            text="🎨 كشف اللون",
            width=25,
            command=lambda: self.processor.set_mode("color")
        )
        self.color_btn.pack(pady=5)
        
        self.all_btn = tk.Button(
            self.root,
            text="🌀 تشغيل الكل",
            width=25,
            command=lambda: self.processor.set_mode("all")
        )
        self.all_btn.pack(pady=5)
        
        self.pause_btn = tk.Button(
            self.root,
            text="⛔ إيقاف مؤقت",
            width=25,
            command=lambda: self.processor.set_mode("none")
        )
        self.pause_btn.pack(pady=5)
        
        self.exit_btn = tk.Button(
            self.root,
            text="❌ خروج",
            width=25,
            command=self.quit_app
        )
        self.exit_btn.pack(pady=10)

    def start_video_thread(self):
        self.video_thread = Thread(target=self.processor.run)
        self.video_thread.daemon = True
        self.video_thread.start()

    def quit_app(self):
        self.processor.stop()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
