import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from picamera2 import Picamera2
import numpy as np
import threading, time

# Khởi tạo camera
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (1020, 600)})
picam2.configure(video_config)
picam2.start()

# Biến chia sẻ frame
latest_frame = None
lock = threading.Lock()

# Thread đọc camera
def camera_thread():
    global latest_frame
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        frame = frame[..., ::-1]  # RGB to BGR nếu muốn OpenCV (PIL dùng RGB)
        request.release()
        with lock:
            latest_frame = frame

t = threading.Thread(target=camera_thread, daemon=True)
t.start()

# Tkinter GUI
root = tk.Tk()
root.title("PiCamera2 Tkinter - FPS Display")

label = tk.Label(root)
label.pack()

frame_count = 0
start_time = time.time()
fps = 0.0

# Font PIL cho text
try:
    font = ImageFont.truetype("arial.ttf", 72)
except:
    font = ImageFont.load_default()

def update_frame():
    global frame_count, start_time, fps
    with lock:
        if latest_frame is not None:
            frame = latest_frame.copy()
            # Vẽ FPS lên frame
            draw = ImageDraw.Draw(Image.fromarray(frame))
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            frame_count += 1
            # Chuyển sang PIL Image
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), f"FPS: {fps:.2f}", font=font, fill=(255, 0, 0))
            image_tk = ImageTk.PhotoImage(image=image)
            label.imgtk = image_tk
            label.configure(image=image_tk)

    root.after(10, update_frame)

update_frame()
root.mainloop()
