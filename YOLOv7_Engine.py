import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path,  non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class YOLOv7():
    def __init__(self, weights:str="yolov7.pt", device:str='', 
                 imgsz:int=1280, conf_thres:float=0.05, iou_thres:float=0.65, *args, **kwargs):
        
        # Define the confidence and IOU thresholds for object detection
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Initialize the logger and CUDA backend for Torch
        set_logging()
        cudnn.benchmark = True
        self.device = select_device(device)   # Select the device for inference (CPU or GPU)
        self.half = self.device.type != 'cpu'  # Enable half-precision (FP16) inference if running on a GPU

        # Load the YOLOv7 model
        self.weights = weights
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # Determine the model stride and set it to an integer value
        
        if self.half:
            self.model.half()  # Convert the model to half-precision (FP16) if running on a GPU

        # Get the names and colors for each object class
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        # Initialize instance variables for the input source, image size, and run flag
        self.imgsz = check_img_size(int(imgsz), s=self.stride)  # check img_size
        
        # Run inference
        self.augment = False
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

    def detect(self, frame):
        # Padded resize
        img = letterbox(frame, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]
                
        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0].detach()

        # Apply NMS
        if "pose" in self.weights:   # pose model
            output = non_max_suppression_kpt(pred, self.conf_thres, self.iou_thres, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
        else: # detection model
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):  #only when it's necessary, i.e., when there are detections
                im0 = frame.copy()
            if len(det) and (not "pose" in self.weights): #  excute object detection
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
            
            elif len(det) and ("pose" in self.weights): #  excute pose estimation                 
                im0 = letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0] 
                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
        
        return im0