
import argparse
import os
import datetime
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import pandas as pd
from jtop import jtop
import mlflow
import wandb

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path,  non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.logging import get_jetson_information, save_summary 

# -------------- helper functions for logging ------------
# Save log file
def log_exp_metric(save_dir:str):
    log_file = save_dir/"log_metric.txt"
    with open(log_file, "w") as log_file:
        log_file.write("Inference, NMS, Plot, Latency, FPS\n")
        for i in range(len(inference_times)):
            log_file.write(f"{inference_times[i]:.2f}, {nms_times[i]:.2f}, {plot_times[i]:.2f}, {latencies[i]:.2f}, {fps_times[i]:.2f}\n")

def log_exp_summary():         
    # save Inference_summary
    path_save  = Path(Path(opt.project).parent)/f'Inference_summary.csv'
    save_data = {}
    metrics = log_metrics(display=False)
    try:
        jetson_information
    except NameError:
        jetson_information = {}
    
    save_data.update(metrics)
    save_data.update(jetson_information)
    save_data.update(argparse_log)
    save_data["date"] = start_time
    save_summary(save_data=save_data, path_save=path_save)

    # Log  
    try:
        with mlflow.start_run():
            mlflow.log_params(jetson_information)
            mlflow.log_params(argparse_log)
            mlflow.log_metrics(metrics)
        wandb_logger.log(metrics)
        wandb_logger.log(jetson_information)
    except Exception as e:
        print(f"An error occurred while logging : {e}")
        
def resize_demo_imgs(img:np.ndarray=None , scale_size:int = 480):
    height, width = img.shape[:2]
    scale = scale_size / max(height, width)
    new_width, new_height = int(width * scale), int(height * scale)             
    resized_image = cv2.resize(img, (new_width, new_height))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
    return resized_image


def log_metrics(display:bool=True)->dict:
    metrics = {"Inference" : np.round(np.mean(inference_times), 2),
        "Latency" : np.round(np.mean(latencies), 2),
        "FPS" : np.round(np.mean(fps_times), 2),
        "Inf_std" : np.round(np.std(inference_times), 2),
        "Lat_std" : np.round(np.std(latencies), 2),
        "FPS_std" : np.round(np.std(fps_times), 2),
        "Initiate_time" : np.round(initiate_times[0], 2)
        }
    if display:
        print(f"\tInitiate_time : {metrics['Initiate_time']:.2f} s")
        print(f"\tInference: {metrics['Inference']:.2f} ms")
        print(f"\tLatency: {metrics['Latency']:.2f} ms")
        print(f"\tFPS: {metrics['FPS']:.2f} frames/s")
    return metrics


# -------------- helper functions for logging ------------
     
def detect(save_img=False):
    initiate_time = time.time()
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size 
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    detect_start_time = time.time()
    initiate_times.append(detect_start_time - initiate_time)
    print(f'\nInitiate_time : {initiate_times[0]:.1f} s')
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        inference_start_time = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0].detach()

        inference_end_time = time_synchronized()

        # Apply NMS
        if "pose" in weights[0]:   # pose model
            output = non_max_suppression_kpt(pred, opt.conf_thres, opt.iou_thres, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
        else: # detection model
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        nms_end_time = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count 
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0) # im0s.shape: (h, w, c)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            
            
            if len(det) and (not "pose" in weights[0]): #  excute object detection 
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            
            elif len(det) and ("pose" in weights[0]): #  excute pose estimation                 
                im0 = letterbox(im0, imgsz, stride=stride, auto=True)[0]
                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
            plot_end_time = time_synchronized()
            
            # Log time (inference + NMS + Plot)
            inference_time = 1E3 * (inference_end_time - inference_start_time)
            nms_time = 1E3 * (nms_end_time - inference_end_time)
            plot_time = 1E3 * (plot_end_time - nms_end_time)
            inference_times.append(inference_time)
            nms_times.append(nms_time)
            plot_times.append(plot_time)
            
            # Calculate latency and FPS and record them
            latency = inference_time + nms_time + plot_time
            fps_ = 1E3 * (1 / latency)
            latencies.append(latency)
            fps_times.append(fps_)
            time_process = f'{s}Done. ({inference_time:.1f}ms) Inference, ({nms_time:.1f}ms) NMS, ({plot_time:.1f}ms) Plot'
            time_process += f' | FPS : { fps_ : .1f} , Latency : {latency:.1f}ms'
            print(time_process)
            
            # Stream results
            # if view_img:
            if dataset.mode != 'image': # 'video' or 'stream'
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                    
                    if opt.log_exp:
                        resized_img = resize_demo_imgs(im0, scale_size=960)
                        caption = f"input size : {opt.img_size} Conf : {opt.conf_thres:.2f}, IOU_thes : {opt.iou_thres:.2f}_{p.stem}"
                        wandb_logger.log({'demo imgs': wandb.Image(resized_img, caption=caption)})
                        # mlflow.log_image(resized_img, f'{caption}.png')
                    
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w = im0.shape[1]
                            h = im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - detect_start_time:.3f}s)')
    log_exp_metric(save_dir=save_dir)
    log_metrics(display=True)
    
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--log_exp', action='store_true', help='log Experimental metrics')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    # --------------Initialize logging------------
    initiate_times, inference_times, nms_times, plot_times,latencies, fps_times  = [], [], [], [], [], []
    argparse_log = vars(opt)    # save argparse.Namespace into dictionary
    try:
        jetson_information = get_jetson_information() # get jetson information 
    except Exception as e:
        print(f"Jetson setting error: {e}")
    
    if opt.log_exp:
        try:
            # mlflow.set_tracking_uri("file:/home/yunghui/experiments/mlruns")
            project_name = "Jetson_Infernece_time_Test"
            # mlflow.create_experiment(project_name)
            mlflow.set_experiment(project_name)
            wandb_logger = wandb.init(project=project_name, resume='allow')
            wandb_logger.config.update(argparse_log)
        except Exception as e:
            print(f"Initialize logging : {e}")
    # --------------Initialize logging------------
    
    
    try:
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov7.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()
        if opt.log_exp:
            log_exp_summary()
    except KeyboardInterrupt:
        if opt.log_exp:
            log_exp_summary()
        