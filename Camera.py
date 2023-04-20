# This code performs real-time object detection using YOLOv7 and a webcam or video file as input.
# To run the code, specify the input source (webcam or video file) and the desired image size (e.g., 416, 512, etc.)

import time
import cv2
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from YOLOv7_Engine import YOLOv7

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path,  non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Camera_YOLOv7(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    
    # Define the path to the YOLOv7 weights file
    weights = ["yolov7-w6-pose.pt"]  # "yolov7-w6-pose.pt", "yolov7.pt"
    
    # Define the confidence and IOU thresholds for object detection
    conf_thres = 0.15
    iou_thres = 0.65

    # Initialize the logger and CUDA backend for Torch
    set_logging()
    cudnn.benchmark = True
    device = select_device('')   # Select the device for inference (CPU or GPU)
    half = device.type != 'cpu'  # Enable half-precision (FP16) inference if running on a GPU

    # Load the YOLOv7 model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # Determine the model stride and set it to an integer value
    
    if half:
        model.half()  # Convert the model to half-precision (FP16) if running on a GPU

    # Get the names and colors for each object class
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    def __init__(self, source:str=None, imgsz:int=640, parent=None, *args, **kwargs):
        super().__init__(parent)
        
        # Initialize instance variables for the input source, image size, and run flag
        self.runFlag = True
        self.runFlag = True
        self.source = source
        self.imgsz = check_img_size(int(imgsz), s=self.stride)  # check img_size
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1
        
        # Set Dataloader
        if self.source.isnumeric() or self.source.endswith(".txt") or self.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
            self.webcam = True
            view_img = check_imshow()
            self.dataset = LoadStreams(
                self.source, img_size=self.imgsz, stride=self.stride)
        else:
            self.webcam = False
            self.dataset = LoadImages(
                self.source, img_size=self.imgsz, stride=self.stride)

    def __del__(self):
        self.runFlag = False
        cv2.destroyAllWindows()
        self.wait()

    def run(self):
        t0 = time.time()
        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img)[0].detach()

            t2 = time_synchronized()

            # Apply NMS
            if "pose" in self.weights[0]:   # pose model
                output = non_max_suppression_kpt(
                    pred, self.conf_thres, self.iou_thres, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
            else:  # detection model
                pred = non_max_suppression(
                    pred, self.conf_thres,  self.iou_thres, )
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                    ), self.dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(
                        self.dataset, 'frame', 0)

                # excute object detection
                if len(det) and (not "pose" in self.weights[0]):
                    # normalization gain whwh
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label,
                                     color=self.colors[int(cls)], line_thickness=1)

                # excute pose estimation
                elif len(det) and ("pose" in self.weights[0]):
                    im0 = letterbox(im0, self.imgsz,
                                    stride=self.stride, auto=True)[0]
                    for idx in range(output.shape[0]):
                        plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
                t4 = time_synchronized()
                 
                
                # convert to QImage
                rgb_image= cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimage = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # qimage_scaled = qimage.scaled(self.imgsz[0], self.imgsz[1], aspectRatioMode=1)
                self.change_pixmap_signal.emit(qimage)
                
                
                # Delay to match the frame rate
                # frame_time = time.time() - t0
                # delay_time = max(1.0 / 60 - frame_time, 0)
                # cv2.waitKey(int(delay_time * 1000))
                
                t5 = time.time()
                t_Latency = (t5-t0)   # Latency
                
                QThread.msleep(10)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object)

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        # self.yolo = YOLOv7(img-size=1280, weights='yolov7.pt')

    def run(self):
        cap = cv2.VideoCapture(self.video_path)

        while True:
            ret, frame = cap.read()
            if ret:
                # pred_frame = self.yolo.predict(frame)
                # self.change_pixmap_signal.emit(pred_frame)
                self.change_pixmap_signal.emit(frame)
            else:
                break
