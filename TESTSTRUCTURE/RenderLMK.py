import cv2
import numpy as np

def render(frame, result, points,p1,p2, color, rad, thick):
    if result.multi_face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in result.multi_face_landmarks:
            # Vẽ các điểm landmark được chọn
            for idx in points:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), rad, color, thick)
                cv2.line(frame,p1,p2,(0,0,255),rad)
                print(p2)
    return frame
