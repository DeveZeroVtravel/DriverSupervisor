import cv2
import dlib
import time
from picamera2 import Picamera2

# Khởi tạo camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1640, 960)})  # Đặt độ phân giải
picam2.configure(config)
picam2.start()

# Load model nhận diện khuôn mặt và landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Biến tính FPS
prev_time = time.time()

# Tạo cửa sổ toàn màn hình để loại bỏ thanh công cụ
cv2.namedWindow("Face Landmark Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Face Landmark Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Các kết nối giữa các điểm landmarks
LANDMARK_CONNECTIONS = [
    # Khuôn mặt ngoài
    list(range(0, 16)),
    # Lông mày trái, phải
    list(range(17, 22)), list(range(22, 27)),
    # Mũi
    list(range(27, 31)), list(range(31, 36)),
    # Mắt trái, phải
    list(range(36, 42)), list(range(42, 48)),
    # Miệng ngoài, miệng trong
    list(range(48, 60)), list(range(60, 68))
]

while True:
    # Chụp ảnh từ camera
    frame = picam2.capture_array()
    
    # Chuyển đổi từ RGB sang BGR để hiển thị đúng màu
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Chuyển đổi sang ảnh xám để nhận diện nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (640, 375))  # Giảm kích thước để tăng tốc phát hiện

    # Lấy độ phân giải
    resolution = picam2.camera_configuration()["main"]["size"]

    # Phát hiện khuôn mặt (dùng ảnh nhỏ để tăng tốc)
    faces = detector(small_gray)
    
    # Tính lại tọa độ khuôn mặt trên ảnh gốc
    scale_x = resolution[0] / 640
    scale_y = resolution[1] / 375

    for face in faces:
        # Chuyển tọa độ về ảnh gốc
        face = dlib.rectangle(
            int(face.left() * scale_x), int(face.top() * scale_y),
            int(face.right() * scale_x), int(face.bottom() * scale_y)
        )
        
        landmarks = predictor(gray, face)

        # Lưu danh sách điểm landmarks
        points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        # Vẽ các đường nối giữa các điểm landmarks
        for connection in LANDMARK_CONNECTIONS:
            for i in range(len(connection) - 1):
                cv2.line(frame, points[connection[i]], points[connection[i + 1]], (255, 255, 255), 1)

        # Vẽ các điểm landmarks
        for (x, y) in points:
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

    # Tính FPS
    curr_time = time.time()
    fps = 2 / (curr_time - prev_time) if prev_time != 0 else 0  # Tránh lỗi chia cho 0
    prev_time = curr_time

    # Hiển thị FPS và độ phân giải trên khung hình
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Resolution: {resolution[0]}x{resolution[1]}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("Face Landmark Detection", frame)

    # Nhấn "q" để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
