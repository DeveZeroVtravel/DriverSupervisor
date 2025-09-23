# yolo_mediapipe_pi.py (optimized)
import cv2
import time
import threading
from picamera2 import Picamera2
import mediapipe as mp
import numpy as np

# ----------------------- SETTINGS -----------------------
YOLO_CFG = "TEST_LANDMARK_MODEL/YOLO_LIB/TINY_FACES/yolov4-tiny-3l.cfg"
YOLO_WEIGHTS = "TEST_LANDMARK_MODEL/YOLO_LIB/TINY_FACES/yolov4-tiny-3l_best.weights"
YOLO_CLASSES = "TEST_LANDMARK_MODEL/YOLO_LIB/face.names"   # chỉ có 1 dòng: face

USE_CUSTOM_FACE_DETECTOR = True
YOLO_INPUT_WIDTH = 255
YOLO_INPUT_HEIGHT = 255
YOLO_CONF_THRESHOLD = 0.2
YOLO_NMS_THRESHOLD = 0.8
MAX_FACE_ROIS = 1
YOLO_INTERVAL = 1  

CAPTURE_SIZE = (1020, 600)
# ---------------------------------------------------------

# Init Picamera2
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"format": "BGR888", "size": CAPTURE_SIZE})
picam2.configure(video_config)
picam2.start()

latest_frame = None
lock = threading.Lock()
stopped = False

def camera_thread():
    global latest_frame, stopped
    while not stopped:
        req = picam2.capture_request()
        frame = req.make_array("main")
        req.release()
        with lock:
            latest_frame = frame

t = threading.Thread(target=camera_thread, daemon=True)
t.start()

# ------------------ Load YOLO (OpenCV DNN) ------------------
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classes = []
try:
    with open(YOLO_CLASSES, "r") as f:
        classes = [c.strip() for c in f.readlines()]
except Exception:
    classes = ["face"]

layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ------------------ Init MediaPipe Face Mesh ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,   # giữ nguyên refine_landmarks
    min_detection_confidence=0.1,
    min_tracking_confidence=0.99
)

# ------------------ Helper ------------------
def run_yolo_on_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0,
                                 (YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    H, W = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            if len(scores) == 0:
                continue
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf > YOLO_CONF_THRESHOLD:
                cx, cy, w, h = int(detection[0]*W), int(detection[1]*H), int(detection[2]*W), int(detection[3]*H)
                x, y = int(cx - w/2), int(cy - h/2)
                boxes.append([x, y, w, h])
                confidences.append(conf)
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONF_THRESHOLD, YOLO_NMS_THRESHOLD)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            results.append((x, y, w, h, confidences[i], class_ids[i] if classes else -1))
    return results

# ------------------ Main loop ------------------
frame_count = 0
start_time = time.time()
fps = 0.0

frame_id = 0
roi_box = None  # lưu box YOLO cuối cùng

try:
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        orig_h, orig_w = frame.shape[:2]
        small = cv2.resize(frame, (YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT))
        scale_x, scale_y = orig_w / YOLO_INPUT_WIDTH, orig_h / YOLO_INPUT_HEIGHT

        frame_id += 1
        detections = []

        # chỉ chạy YOLO mỗi YOLO_INTERVAL frame
        if frame_id % YOLO_INTERVAL == 0 or roi_box is None:
            detections = run_yolo_on_frame(small)
            detections = sorted(detections, key=lambda r: r[4], reverse=True)[:MAX_FACE_ROIS]
            if detections:
                roi_box = detections[0]
        else:
            if roi_box is not None:
                detections = [roi_box]

        if detections:
            for (x_s, y_s, w_s, h_s, conf, class_id) in detections:
                x, y, w, h = int(x_s*scale_x), int(y_s*scale_y), int(w_s*scale_x), int(h_s*scale_y)
                x, y = max(0, x), max(0, y)
                w, h = min(w, orig_w-x), min(h, orig_h-y)

                # Thu nhỏ box (shrink ratio)
                shrink_ratio = 0.9
                new_w = int(w * shrink_ratio)
                new_h = int(h * shrink_ratio)
                x = x + (w - new_w) // 2
                y = y + (h - new_h) // 2
                w, h = new_w, new_h

                label = f"{conf:.2f}"
                if classes and 0 <= class_id < len(classes):
                    label = f"{classes[class_id]}:{conf:.2f}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, max(15, y-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_roi)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for lm in face_landmarks.landmark:
                            px, py = int(lm.x * w) + x, int(lm.y * h) + y
                            cv2.circle(frame, (px, py), 1, (255, 255, 0), -1)
                break

        # --- FPS ---
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLO + MediaPipe (Picamera2)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stopped = True
    t.join(timeout=1.0)
    cv2.destroyAllWindows()
    face_mesh.close()
    picam2.stop()
