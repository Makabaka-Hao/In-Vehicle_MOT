import argparse
import os
import platform
import shutil
import pathlib
import time
from pathlib import Path
import sys
import json
import os
import cv2
import numpy as np
from collections import deque
from track_main import pipeline
from glob import glob
sys.path.insert(1, '/project/train/src_repo/yolov5/')
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
sys.path.append('../../github/ByteTrack/')
from yolox.tracker.byte_tracker import BYTETracker, STrack

# ####### 参数设置
conf_thres = 0.7
iou_thres = 0.3
prob_thres = 0.5
#######
imgsz = [1024,1024]
weights = "/project/train/models/exp2/weights/best.pt"
device = '0'
stride = 32
names = ["car", "SUV", "passenger_car", "truck", "van", "slagcar", "bus", "ambulance", "fire_truck", "police_car", "engineering_rescue_vehicle", "microbus", "tractor", "other_vehicle"]
def init():
    # Initialize
    global imgsz, device, stride
    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model.half()  # to FP16
    model.eval()
    # model.warmup(imgsz=(1, 3, 480, 480))  # warmup
    car_detector = CarDetectot_SSD()    # 生成基于SSD的车辆检测器
    return model

def process_video(handle=None, input_video=None, args=None, ** kwargs):
    global imgsz
    
    args = json.loads(args)
    output_tracker_file = args['output_tracker_file']
    # output_tracker_file = "test.txt"
    #print("file_name:",output_tracker_file)
    frames_dict = {}

    for frame in pathlib.Path(input_video).glob('*.jpg'):

        frame_id = int(frame.with_suffix('').name)

        frames_dict[frame_id] = frame.as_posix()

    frames = list(frames_dict.items())  # frames[¨] = (frame_id, frame_file)
    frame_count = len(frames)
    
    result = None
    
    for frame_id, frame_file in frames:
        img0 = cv2.imread(frame_file)
    
        # Padded resize

        img = letterbox(img0, new_shape=imgsz, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img/255  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0).to(device)
        img = img.type(torch.cuda.HalfTensor)
    
    
    #     # Convert
    #     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    #     img = np.ascontiguousarray(img)

    #     img = torch.from_numpy(img).to(device)
    #     img = img.half()
    # #     img = img.float()
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if len(img.shape) == 3:
    #         img = img[None]
        pred = handle(img, augment=False, val=True)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

        fake_result = {}

        fake_result["algorithm_data"] = {
           "is_alert": False,
           "target_count": 0,
           "target_info": []
       }
        fake_result["model_data"] = {"objects": []}
        # Process detections
        cnt = 0
        bbox = []
        clses = []
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            print(len(det))
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in det:
                    if conf < prob_thres:
                        continue
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    bbox.append([xmin, ymin, xmax, ymax])
                    clses.append(int(cls))
                


        @dataclass(frozen=True)
        class BYTETrackerArgs:
            track_thresh: float = 0.25
            track_buffer: int = 30
            match_thresh: float = 0.8
            aspect_ratio_thresh: float = 3.0
            min_box_area: float = 1.0
            mot20: bool = False

        byte_tracker = BYTETracker(BYTETrackerArgs())

        tracks = byte_tracker.update(
                output_results=np.array(bbox, dtype=float),
                img_info=frame.shape,
                img_size=frame.shape
            )

        for i in range(len(id)):
            fake_result["model_data"]['objects'].append({
                        "x":x[i],
                        "y":y[i],
                        "width":width[i],
                        "height":height[i],
                        "vechile_type":names[vechile_type[i]],
                        "confidence_type":0.9892,
                        "vechile_color":"white",
                        "confidence_colore":0.9892,
                        "id":id[i],
                    })
            cnt+=1
            fake_result["algorithm_data"]["target_info"].append({
                        "x":x[i],
                        "y":y[i],
                        "width":width[i],
                        "height":height[i],
                        "vechile_type":names[vechile_type[i]],
                        "confidence_type":0.9892,
                        "vechile_color":"white",
                        "confidence_colore":0.9892,
                        "id":id[i],
                    })

        if cnt>0:
            fake_result ["algorithm_data"]["is_alert"] = True
            fake_result ["algorithm_data"]["target_count"] = cnt
        else:
            fake_result ["algorithm_data"]["target_info"]=[]

   
        with open(output_tracker_file, 'a+') as tracker_file:
            for i in range(len(id)):
                text = str(frame_id) + "," + str(id[i]) + "," + str(x[i]) + "," + str(y[i]) + "," + str(width[i]) + "," + str(height[i]) + ",1," + str(vechile_type[i] + 1) + ",1\n"
                #print("text:",text)
                tracker_file.write(text)
    
        result = fake_result

    return json.dumps(result, indent = 4)

# if __name__ == '__main__':
# #     from glob import glob
# #     # Test API
# #     image_names = glob('/home/data/2850/*.jpg')
#     predictor = init()
#     res = process_video(predictor, '/home/data/2850')
#     res = process_video(predictor, '/home/data/2851')
#     s = 0
#     for image_name in image_names:
#         print(image_name)
#         img = cv2.imread(image_name)
#         t1 = time.time()
#         res = process_video(predictor, img)
#         print(res)
#         t2 = time.time()
#         s += t2 - t1

#     print(1/(s/100))