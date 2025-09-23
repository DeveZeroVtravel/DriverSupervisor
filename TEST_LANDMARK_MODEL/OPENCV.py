import cv2
import dlib
from picamera2 import Picamera2
import threading, time

# --- Camera setup ---
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"format": "BGR888", "size": (1020, 600)})
picam2.configure(video_config)
picam2.start()

# --- Dlib detector + predictor ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("TEST_LANDMARK_MODEL/shape_predictor_68_face_landmarks.dat")

# --- Shared variables ---
latest_frame = None      # raw frame từ camera
processed_frame = None   # frame có landmark
lock_in = threading.Lock()
lock_out = threading.Lock()
cam_fps=0

# --- FPS counter ---
cam_frames = 0
lmk_frames = 0
start_time = time.time()
cam_fps=0

# --- Scale factor ---
SCALE = 0.3  # giảm kích thước xuống 50% để tăng tốc

# --- Thread đọc camera ---
def camera_thread():
    global latest_frame, cam_frames
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        request.release()
        with lock_in:
            latest_frame = frame
        cam_frames += 1

# --- Thread xử lý landmark ---
def landmark_thread():
    global latest_frame, processed_frame, lmk_frames
    while True:
        with lock_in:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Resize để tăng tốc
        small_frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect trên ảnh nhỏ
        faces = detector(gray, 0)

        for face in faces:
            # Landmark trên ảnh nhỏ
            shape = predictor(gray, face)
            for i in range(68):
                part = shape.part(i)
                # Scale lại lên ảnh gốc
                px, py = int(part.x / SCALE), int(part.y / SCALE)
                cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

        with lock_out:
            processed_frame = frame
        lmk_frames += 1

# --- Start threads ---
t_cam = threading.Thread(target=camera_thread, daemon=True)
t_lmk = threading.Thread(target=landmark_thread, daemon=True)
t_cam.start()
t_lmk.start()

# --- Main loop ---
while True:
    with lock_out:
        if processed_frame is not None:
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                cam_fps = cam_frames / elapsed
                lmk_fps = lmk_frames / elapsed
                print(f"Camera FPS: {cam_fps:.2f} | Landmark FPS: {lmk_fps:.2f}")
                cam_frames = 0
                lmk_frames = 0
                start_time = time.time()
            cv2.putText(processed_frame, f"FPS: {cam_fps:.2f}", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.imshow("PiCamera2 + dlib68 (Scale Down)", processed_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
