from LandmarksList import DIRP
from IData import setting
import cv2
import math
import numpy as np

def estHeadPose(frame,res):
    if not res.multi_face_landmarks:
        return None,None,None,None,None
    
    h=setting["frameRes"]["h"]
    w=setting["frameRes"]["w"]

    face2d, face3d=[], []
    for face_landmarks in res.multi_face_landmarks:
        for idx in DIRP:
            lm = face_landmarks.landmark[idx]
            
            x,y=int(lm.x*w), int(lm.y*h)
            face2d.append([x,y])
            face3d.append([x,y,lm.z])
            if idx == 1:
                nose2d=(x,y)
                nose3d=(x,y,lm.z*3000)
        face2d=np.array(face2d,dtype=np.float64)
        face3d=np.array(face3d,dtype=np.float64)

        if face2d.shape[0] < 4 or face3d.shape[0] < 4:
            return None,None,None,None,None

        cDeg=60.2
        fLenght=w/(2*math.tan(math.radians(cDeg/2)))
        cx,cy=w/2,h/2
        camMatrix=np.array([
            [fLenght, 0      , cx],
            [0      , fLenght, cy],
            [0      , 0      , 1 ]
        ],dtype=np.float64)

        distMatrix= np.zeros((4,1),dtype=np.float64)


        _, rotVec,transVec=cv2.solvePnP(face3d,face2d,camMatrix,distMatrix)
        rMatrix,_=cv2.Rodrigues(rotVec)
        angles,_,_,_,_,_=cv2.RQDecomp3x3(rMatrix)

        pitch, yaw, roll =[math.degrees(a)for a in angles]

        p1=(int(nose2d[0]),int(nose2d[1]))
        p2=(int(nose2d[0]+yaw*20),int(nose2d[1]-pitch*20))

        return yaw,pitch,roll, p1, p2
