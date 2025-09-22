import mediapipe as mp

mp_face_mesh=mp.solutions.face_mesh
faceMesh=mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.22,
    min_tracking_confidence=0.78
)