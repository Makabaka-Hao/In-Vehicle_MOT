import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import pathlib
import sys
import json
import os
import cv2
import numpy as np
from collections import deque
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
from PIL import Image
from AIDetector_pytorch import Detector
import re

# ####### 参数设置
conf_thres = 0.5
iou_thres = 0.5
prob_thres = 0.5
#######
imgsz = 480
weights = "/project/train/models/exp/weights/best.pt"
device = '0'
stride = 32
names = ["car", "SUV", "passenger_car", "truck", "van", "slagcar", "bus", "ambulance", "fire_truck", "police_car", "engineering_rescue_vehicle", "microbus", "tractor", "other_vehicle"]
frame_id = 0
def init():
    # # Initialize

    # Load model
    det = Detector()

    return det

def process_video(model, input_video=None, args=None, **kwargs):
    model = Detector()
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
#         frame_id  = int(re.findall(r'\d+', str(img_path))[1])
#         # print(frame_id)

#         #print("path:",img_path)
#         img0 = Image.open(img_path)
#         img0 = np.array(img0)
        #print("img:",type(img0))
        result = model.feedCap(img0)

        result = result['list_of_ids']
        id = [item[0] for item in result]
        # print(id)
        
        # print(result)
        x = [item[1] for item in result]
        y = [item[2] for item in result]
        width = [item[3] for item in result]
        height = [item[4] for item in result]
        cls = [int(item[5][5]) for item in result]
        
        fake_result = {}

        fake_result["algorithm_data"] = {
           "is_alert": False,
           "target_count": 0,
           "target_info": []
       }
        fake_result["model_data"] = {"objects": []}


        for i in range(len(id)):
            fake_result["model_data"]['objects'].append({
                        "x":x[i],
                        "y":y[i],
                        "width":width[i],
                        "height":height[i],
                        "vechile_type":names[cls[i]],
                        "confidence_type":0.9892,
                        "vechile_color":"white",
                        "confidence_colore":0.9892,
                        "id":id[i],
                    })
            
            fake_result["algorithm_data"]["target_info"].append({
                        "x":x[i],
                        "y":y[i],
                        "width":width[i],
                        "height":height[i],
                        "vechile_type":names[cls[i]],
                        "confidence_type":0.9892,
                        "vechile_color":"white",
                        "confidence_colore":0.9892,
                        "id":id[i],
                    })

        # if cnt>0:
        #     fake_result ["algorithm_data"]["is_alert"] = True
        #     fake_result ["algorithm_data"]["target_count"] = cnt
        # else:
        #     fake_result ["algorithm_data"]["target_info"]=[]

        # args = json.loads(args)
        # output_tracker_file = args['output_tracker_file']
        # # output_tracker_file = "/project/ev_sdk/src/test1.txt"
        with open(output_tracker_file, 'a+') as tracker_file:
            # print("len:",len(id))
            for i in range(len(id)):
                text = str(frame_id) + "," + str(id[i]) + "," + str(x[i]) + "," + str(y[i]) + "," + str(width[i]) + "," + str(height[i]) + ",1," + str(cls[i] + 1) + ",1\n"
                # print("text:",text)
                tracker_file.write(text)
            # tracker_file.write("***************************\n")
    
        result = fake_result

    return json.dumps(result, indent = 4)

# if __name__ == '__main__':
# #     from glob import glob
# #     # Test API
# #     image_names = glob('/home/data/2850/*.jpg')
#     predictor = init()
#     res = process_video(predictor, '/home/data/2850')
#     s = 0
#     for image_name in image_names:
#         print(image_name)
#         img = cv2.imread(image_name)
#         t1 = time.time()
#         res = process_image(predictor, img)
#         print(res)
#         t2 = time.time()
#         s += t2 - t1

#     print(1/(s/100))