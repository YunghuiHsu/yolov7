
## Pose estimation

[`code`](https://github.com/WongKinYiu/yolov7/tree/pose) [`yolov7-w6-pose.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

See [keypoint.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/keypoint.ipynb).

<div align="center">
    <a href="./">
        <img src="./figure/pose.png" width="39%"/>
    </a>
</div>


Add support for "pose estimation" from the official YOLOv7 repository, can simply load ` yolov7-w6-pose.pt` weights directly into the Pytorch environment for pedestrian skeleton detection

`detect_pose.py` modified from `detect.py`, adding support for "pose estimation" to the original `detect.py` architecture, without modifying, adding any `utils` files. Directly download and ready to use

For environment setup and package installation, please refer to the official YOLOv7 repository


###  Inference for Pose estimation
The parameters `conf` and `iou-thres` need to be tested
- On the video:
```
python detect_pose.py --weights yolov7-w6-pose.pt --conf 0.25 --iou-thres 0.65 --img-size 640 --source yourvideo.mp4 --no-trace
```

- On the image:
```
python detect_pose.py --weights yolov7-w6-pose.pt --conf 0.25 --iou-thres 0.65 --img-size 640 --source inference/images/horses.jpg --no-trace
```

- On the webcam:  
    add `0` after --source
```
python detect_pose.py --weights yolov7-w6-pose.pt --conf 0.25 --iou-thres 0.65 --img-size 640 --source 0 --no-trace
```



## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```
