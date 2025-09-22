import cv2
import time
import threading
from picamera2 import Picamera2
import mediapipe as mp
import numpy as np

# --- Bật OpenCL nếu có ---
cv2.ocl.setUseOpenCL(True)

# --- Camera setup ---
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"format": "BGR888", "size": (1020, 600)}
)
picam2.configure(video_config)
picam2.start()

# --- Mediapipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.7
)

# --- Landmark cần vẽ ---
EYE_LEFT = [246,161,160,159,158,157,173,33,7,163,144,145,153,154,155,133]
EYE_RIGHT = [466,388,387,386,385,384,398,263,249,390,373,374,380,381,382,362]
NOSE = [168,6,197,195,5,4,1,19,94,2]
MOUTH = [78,191,80,81,82,13,312,311,310,415,308,78,95,88,178,87,14,317,402,318,324,308]
INTEREST_POINTS = EYE_LEFT + EYE_RIGHT + NOSE + MOUTH

# --- Biến chia sẻ frame ---
latest_frame = None
lock = threading.Lock()
fps = 0

# --- Thread đọc camera ---
def camera_thread():
    global latest_frame
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        request.release()
        with lock:
            latest_frame = frame

t = threading.Thread(target=camera_thread, daemon=True)
t.start()

# --- Thiết lập full screen ---
cv2.namedWindow("PiCamera2 + Mediapipe", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("PiCamera2 + Mediapipe", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Vòng lặp chính ---
frame_count = 0
start_time = time.time()

while True:
    with lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    h, w, _ = frame.shape

    # --- Resize nhỏ hơn để Mediapipe chạy nhanh hơn ---
    small_frame = cv2.resize(frame, (320, 180))
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Mediapipe xử lý
    results = face_mesh.process(rgb_small)

    # Vẽ landmark mắt, mũi, miệng
    if results.multi_face_landmarks:
        scale_x = w / 320
        scale_y = h / 180
        for face_landmarks in results.multi_face_landmarks:
            for idx in INTEREST_POINTS:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * 320 * scale_x), int(lm.y * 180 * scale_y)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("PiCamera2 + Mediapipe", frame)

    # FPS
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()