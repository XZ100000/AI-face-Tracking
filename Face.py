import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

class FaceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")

        self.video_source = 0  # Change this if you have a different video source

        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_start = tk.Button(root, text="Start", command=self.start_video)
        self.btn_start.pack(pady=10)

        self.btn_stop = tk.Button(root, text="Stop", command=self.stop_video)
        self.btn_stop.pack(pady=5)

        self.delay = 10
        self.update()

        self.root.mainloop()

    def start_video(self):
        if not self.vid.isOpened():
            self.vid.open(self.video_source)

    def stop_video(self):
        if self.vid.isOpened():
            self.vid.release()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(self.delay, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectorApp(root)
