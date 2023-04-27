# This code performs real-time object detection using YOLOv7 and a webcam or video file as input.
# To run the code, specify the input source (webcam or video file) and the desired image size (e.g., 416, 512, etc.)

import time
import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage

from YOLOv7_Engine import YOLOv7


class LiveFrameThread(QThread):
    update_signal = pyqtSignal(QImage)

    def __init__(self, source: str = None, imgsz: int = 640, conf_thres=0.05, iou_thres=0.65, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ThreadActive = True
        self.source = source
        self.model = YOLOv7(weights="yolov7.pt", imgsz=imgsz,
                            conf_thres=conf_thres, iou_thres=conf_thres)  # default

    def start_capture(self):
        self.capture = cv2.VideoCapture(self.source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    def stop_capture(self):
        self.capture.release()

    def read_frame(self):
        ret, frame = self.capture.read()
        if ret:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            return frame, h, w, bytes_per_line
        return None, None, None, None

    def convert_to_qimage(self, frame, h, w, bytes_per_line):
         # Convert processed frame to QImage 

        qimage = QImage(frame, w, h, bytes_per_line,
                        QImage.Format_RGB888).rgbSwapped()
        qimage = qimage.scaled(w, h, transformMode=Qt.SmoothTransformation)
        return qimage

    def run(self):
        self.start_capture()
        while self.ThreadActive:
            try:
                # Read frame from input source
                frame, h, w, bytes_per_line = self.read_frame()
                if frame is not None:
                    # Process the frame using YOLOv7 engine
                    processed_frame = self.model.detect(frame)
                    # Convert processed frame to QImage 
                    qimage = self.convert_to_qimage(
                        processed_frame, h, w, bytes_per_line)
                    # Emit the QImage signal
                    self.update_signal.emit(qimage)
                    # Sleep for 10ms to control frame rate 
                    QThread.msleep(10)
            except Exception as e:
                print(str(e))
            # key = cv2.waitKey(100) & 0xFF
            # if key == ord('q'):
            #     break
        self.stop_capture()

    def __del__(self):
        self.ThreadActive = False
        cv2.destroyAllWindows()
        self.wait()
