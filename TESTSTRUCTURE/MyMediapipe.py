import mediapipe as mp
from IData import setting

mp_face_mesh=mp.solutions.face_mesh
faceMesh=mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=setting["mediapipe"]["detectionConfident"],
    min_tracking_confidence=setting["mediapipe"]["trackingConfident"]
)