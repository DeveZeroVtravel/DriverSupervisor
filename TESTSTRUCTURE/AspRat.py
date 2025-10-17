from IData import setting
import numpy as np

w=setting["frameRes"]["w"]
h=setting["frameRes"]["h"]

def AspectRatio(landmarks,points):
    p1,p2,p3,p4,p5,p6=[(int(landmarks[i].x*w),int(landmarks[i].y*h))for i in points]
    res=(np.linalg.norm(np.array(p2)-np.array(p6))+\
         np.linalg.norm(np.array(p3)-np.array(p5)))/\
         (2.0 * np.linalg.norm(np.array(p1)-np.array(p4)))
    return res