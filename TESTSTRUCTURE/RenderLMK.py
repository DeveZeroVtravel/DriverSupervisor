import cv2
def render(frame,result,points,color,rad,thick):
    if result.multi_face_landmarks:
        h,w,_=frame.shape
        for face_landmarks in result.multi_face_landmarks:
            for idx in points:
                lm=face_landmarks.landmark[idx]
                x,y=int(lm.x*w), int(lm.y*h)
                cv2.circle(frame,(x,y),rad,color,thick)
    return frame