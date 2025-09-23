import cv2
from picamera2 import Picamera2
import threading, time

# Khởi tạo camera
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"format": "BGR888", "size": (1280, 720)})
picam2.configure(video_config)
picam2.start()

# Biến chia sẻ frame
latest_frame = None
lock = threading.Lock()
fps=0

# Thread đọc camera
def camera_thread():
    global latest_frame
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ##frame = frame[..., ::-1]
        request.release()
        with lock:
            latest_frame = frame

t = threading.Thread(target=camera_thread, daemon=True)
t.start()

frame_count = 0
start_time = time.time()

while True:
    with lock:
        if latest_frame is not None:
            cv2.putText(latest_frame, f"FPS: {fps:.2f}", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.imshow("PiCamera2 OpenCV - Optimized", latest_frame)

    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        print(f"OpenCV Optimized FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
