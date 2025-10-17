import cv2
import time
import threading
import LandmarksList
import numpy as np
from MyCamera import myCamera
from MyMediapipe import faceMesh
from RenderLMK import render
from AspRat import AspectRatio as AR
from collections import deque
from IData import setting

w=setting["frameRes"]["w"]
h=setting["frameRes"]["h"]


EAR_Q=deque(maxlen=setting["Adjustment"]["AvgEAR"])
ThreshEAR=setting["Adjustment"]["ThreshEAR"]
DurEAR=setting["Adjustment"]["DurEAR"]
blinkCount = 0
closeFrames = 0

MOUTH_Q=deque(maxlen=setting["Adjustment"]["AvgMOUTH"])
ThreshMOUTH=setting["Adjustment"]["ThreshMOUTH"]
DurMOUTH=setting["Adjustment"]["DurMOUTH"]
yawnCount = 0
openFrames = 0

camera=myCamera()

cv2.namedWindow("DSDS",cv2.WINDOW_NORMAL)
cv2.setWindowProperty("DSDS",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

fps,frameCount,StartTime=0,0,time.time()
viewMode=False


while True:
    frame=camera.get_frame()
    if frame is None:
        continue

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=faceMesh.process(frame)
    

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            EAR =(AR(face_landmarks.landmark,LandmarksList.EYE_LC)+\
                  AR(face_landmarks.landmark,LandmarksList.EYE_RC))/2.0
            MOUTH=AR(face_landmarks.landmark,LandmarksList.MOUTH_C)
            
            EAR_Q.append(EAR)
            MOUTH_Q.append(MOUTH)

            if np.mean(EAR_Q)<ThreshEAR:
                closeFrames+=1
            else: 
                if closeFrames>=DurEAR:
                    blinkCount+=1
                closeFrames=0

            if np.mean(MOUTH_Q)>ThreshMOUTH:
                openFrames+=1
            else: 
                
                if openFrames>=DurMOUTH:
                    yawnCount+=1
                openFrames=0

    if viewMode==True:
        frame=render(frame,result,LandmarksList.selectPoint,(0,255,0),1,-1)
        
    cv2.putText(frame, f"FPS:{fps:.2f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.putText(frame, f"BLINK:{blinkCount}",(10,555),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.putText(frame, f"YAWN:{yawnCount}",(10,585),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("DSDS",frame)

    frameCount+=1
    tc=time.time()-StartTime
    if tc>=1:
        fps=frameCount/tc
        frameCount,StartTime=0,time.time()
    
    KB=cv2.waitKey(1) & 0xFF

    match KB:
        case 113:
            break
        case 118:
            viewMode=not viewMode
        case _:
            pass

cv2.destroyAllWindows()