import cv2
import threading
from IData import setting
from picamera2 import Picamera2

#----ReadSettingConfig----
w=setting["frameRes"]["w"]
h=setting["frameRes"]["h"]

#----ObjectiveCamSetup----
class myCamera:
    def __init__(self, size=(w,h)):
        self.picam2=Picamera2()
        video_config=self.picam2.create_video_configuration(main={"format":"BGR888","size":size})
        self.picam2.configure(video_config)
        self.picam2.start()

        self.latest_frame = None
        self.lock=threading.Lock()

        self.thread=threading.Thread(target=self._camera_thread,daemon=True)
        self.thread.start()
    
    def _camera_thread(self):
        while True:
            request=self.picam2.capture_request()
            frame=request.make_array("main")
            request.release()
            with self.lock:
                self.latest_frame=frame.copy()
    
    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()