import cv2
import time
import threading
from MyCamera import myCamera
from MyMediapipe import faceMesh
from LandmarksList import selectPoint
from RenderLMK import render

camera=myCamera()

cv2.namedWindow("DSDS",cv2.WINDOW_NORMAL)
cv2.setWindowProperty("DSDS",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

fps,frameCount,StartTime=0,0,time.time()

while True:
    frame=camera.get_frame()
    if frame is None:
        continue

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=faceMesh.process(frame)
    
    frame=render(frame,result,selectPoint,(0,255,0),1,-1)

    cv2.putText(frame, f"FPS:{fps:.2f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("DSDS",frame)

    frameCount+=1
    tc=time.time()-StartTime
    if tc>=1:
        fps=frameCount/tc
        frameCount,StartTime=0,time.time()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()