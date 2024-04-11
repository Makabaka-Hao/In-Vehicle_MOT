import argparse
import os
import platform
import shutil
import pathlib
import time
from pathlib import Path
import sys
import json
# from ssd_detector import CarDetectot_SSD
import os
import cv2
import numpy as np
from collections import deque
# from ssd_detector import CarDetectot_SSD
from track_main import pipeline
from scipy.optimize import linear_sum_assignment      # sklearn 0.19.1
from kalman_tracker import Kalman_Tracker
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

# ####### 参数设置
conf_thres = 0.4
iou_thres = 0.45
prob_thres = 0.5
#######
imgsz = [640,640]
weights = "/project/train/models/exp/weights/best.pt"
device = '0'
stride = 32
names = ["car", "SUV", "passenger_car", "truck", "van", "slagcar", "bus", "ambulance", "fire_truck", "police_car", "engineering_rescue_vehicle", "microbus", "tractor", "other_vehicle"]


import os
import sys
par_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(par_path)
sys.path.append(os.path.join(par_path, 'yolov5'))
from utils.torch_utils import select_device
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from glob import glob
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox
from bytetrack_tracker.byte_tracker import BYTETracker
from deepsort_tracker.deepsort import DeepSort
from sort_tracker.sort import Sort
from bot_tracker.mc_bot_sort import BoTSORT
from motdt_tracker.motdt_tracker import OnlineTracker

import warnings
warnings.filterwarnings("ignore")

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


class Detect_Track:
    def __init__(self, tracker='ByteTrack', model_path='/project/train/models/exp9/weights/best.pt', imgsz=(640,640), vis=True):
        yolo_model = os.path.join(par_path, model_path)
        self.device = select_device('0')
        self.model = DetectMultiBackend(yolo_model, device=self.device, dnn=False, fp16=True)

        self.names = self.model.names
        self.stride = self.model.stride
        self.imgsz = check_img_size( imgsz, s=self.stride)  # check image size

        self.trt = model_path.endswith('.engine') or model_path.endswith('.trt')

        self.vis = vis

        
        if tracker == 'ByteTrack':
            self.tracker = BYTETracker(track_thresh=0.55, track_buffer=100, match_thresh=0.65)
            self._type = 'ByteTrack'

        elif tracker == 'DeepSort':
            self.tracker = DeepSort('./models/ckpt.t7', max_dist=0.1, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=30, n_init=3, nn_budget=100, use_cuda=True)
            self._type = 'DeepSort'

        elif tracker == 'Sort':
            self.tracker = Sort(det_thresh = 0.7)
            self._type = "Sort"

        elif tracker == "BoTSort":
            self.tracker = BoTSORT(track_high_thresh=0.3, track_low_thresh=0.05, new_track_thresh=0.4, 
                                   match_thresh=0.7, track_buffer=30,frame_rate=30,
                                   with_reid = False, proximity_thresh=0.5, appearance_thresh=0.25,
                                   fast_reid_config=None, fast_reid_weights=None, device=None)
            self._type = "BoTSort"
        
        elif tracker == "motdt":
            self.tracker = OnlineTracker('./models/googlenet_part8_all_xavier_ckpt_56.h5', min_cls_score=0.4, min_ap_dist=0.8, 
                                         max_time_lost=30, use_tracking=True, use_refind=True)
            self._type = "motdt"

        else:
            raise Exception('Tracker must be ByteTrack/DeepSort/Sort/BoTSort/motdt.')

        
    @torch.no_grad()
    def __call__(self, image: np.ndarray):

        img_vis = image.copy()
        
        clss = []
        tlwhs = []
        tids = []

        # Run tracking
        img = letterbox(image, self.imgsz, stride=self.stride, auto= not self.trt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        im = img.half()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]
        
        # inference
        pred = self.model(im, augment=False, visualize=False)
        # Apply NMS
        pred = non_max_suppression(pred, 0.5, 0.5, None, False, max_det=1000)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()
                
                if self._type == 'ByteTrack':
                    online_targets  = self.tracker.update(det[:, :6].cpu(), [image.shape[0], image.shape[1]], self.imgsz)
                    
                    if len(online_targets) > 0:
                        for t in online_targets:

                            clss.append(t.cls.item())
                            tlwhs.append([ t.tlwh[0], t.tlwh[1], t.tlwh[2], t.tlwh[3] ])
                            tids.append(t.track_id)

                            if self.vis:
                                color = get_color(int(t.cls.item())+1)
                                cv2.rectangle(img_vis, (int(t.tlwh[0]), int(t.tlwh[1])), (int(t.tlwh[0]+t.tlwh[2]), int(t.tlwh[1]+t.tlwh[3])), color, 2)
                                cv2.putText(img_vis, self.names[int(t.cls.item())]+'  '+str(t.track_id), (int(t.tlwh[0]),int(t.tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, color)
                
                elif self._type == 'Sort':
                    online_targets  = self.tracker.update(det[:, :6].cpu(), [image.shape[0], image.shape[1]], self.imgsz)
                    if len(online_targets) > 0:
                        for t in online_targets:
                            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                            tid = t[4]
                            cls = t[5]
                            
                            tlwhs.append(tlwh)
                            tids.append(int(tid))
                            clss.append(cls)

                            if self.vis:
                                color = get_color(int(cls)+1)
                                cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), color, 2)
                                cv2.putText(img_vis,  self.names[int(cls)]+'  '+str(int(tid)), (int(tlwh[0]),int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, color)

                elif self._type == "BoTSort":
                    online_targets  = self.tracker.update(det[:, :6].cpu(), image)
                    for t in online_targets:
                        tlwh = t.tlwh
                        tlbr = t.tlbr
                        tid = t.track_id
                        tcls = t.cls
                        
                        tlwhs.append(tlwh)
                        tids.append(int(tid))
                        clss.append(tcls)
                        
                        if self.vis:
                            color = get_color(int(tcls)+1)
                            cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), color, 2)
                            cv2.putText(img_vis,  self.names[int(tcls)]+'  '+str(int(tid)), (int(tlwh[0]),int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, color) 
                
                elif self._type == "DeepSort":
                    online_targets  = self.tracker.update(det[:, :6].cpu(), image)
                    for t in online_targets:
                        tlbr = [t[0], t[1], t[2], t[3]]
                        tlwh = [t[0], t[1], t[2]-t[0], t[3]-t[1]]
                        tid = t[4]
                        tcls = t[5]

                        tlwhs.append(tlwh)
                        tids.append(int(tid))
                        clss.append(tcls)

                        if self.vis:
                            color = get_color(int(tcls)+1)
                            cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), color, 2)
                            cv2.putText(img_vis,  self.names[int(tcls)]+'  '+str(int(tid)), (int(tlwh[0]),int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, color)

                elif self._type == "motdt":
                    online_targets  = self.tracker.update(det[:, :6].cpu(), image)
                    for t in online_targets:
                        tlwh = t.tlwh
                        tlbr = t.tlbr
                        tid = t.track_id
                        tcls = t.cls

                        tlwhs.append(tlwh)
                        tids.append(int(tid))
                        clss.append(tcls)

                        if self.vis:
                            color = get_color(int(tcls)+1)
                            cv2.rectangle(img_vis, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])), color, 2)
                            cv2.putText(img_vis,  self.names[int(tcls)]+'  '+str(int(tid)), (int(tlwh[0]),int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8, color)

        return img_vis, clss, tlwhs, tids



def init():
    # Initialize
#     global imgsz, device, stride
#     set_logging()
#     device = select_device('0')
#     half = device.type != 'cpu'  # half precision only supported on CUDA

#     # Load model
#     model = DetectMultiBackend(weights, device=device, dnn=False)
#     stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
#     imgsz = check_img_size(imgsz, s=stride)  # check img_size
#     model.half()  # to FP16
#     model.eval()
#     # model.warmup(imgsz=(1, 3, 480, 480))  # warmup
#     # car_detector = CarDetectot_SSD()    # 生成基于SSD的车辆检测器
    model = Detect_Track()
    return model

def process_video(handle=None, input_video=None, args=None, ** kwargs):
    model = Detect_Track()
    args = json.loads(args)
    output_tracker_file = args['output_tracker_file']
    # output_tracker_file = "/project/ev_sdk/src/test.txt"
    frames_dict = {}
    for frame in pathlib.Path(input_video).glob('*.jpg'):

        frame_id = int(frame.with_suffix('').name)

        frames_dict[frame_id] = frame.as_posix()

    frames = list(frames_dict.items())  # frames[¨] = (frame_id, frame_file)
    
    for frame_id, frame_file in frames:
        img0 = cv2.imread(frame_file)
        img_vis, clss, tlwhs, tids = model(img0)
        # print("#img:",img_vis)
        # print("#cls:",clss)
        # print("#tlwhs:",tlwhs)
        # print("#tids:",tids)
        
        with open(output_tracker_file, 'a+') as tracker_file:
            for i in range(len(tids)):
                text = str(frame_id) + "," + str(tids[i]) + "," + str(tlwhs[i][0]) + "," + str(tlwhs[i][1]) + "," + str(tlwhs[i][2]) + "," + str(tlwhs[i][3]) + ",1," + str(int(clss[i]) + 1) + ",1\n"
                # print("text:",text)
                tracker_file.write(text)

    fake_result = {}
    fake_result["algorithm_data"] = {
        "is_alert": False,
        "target_count": 0,
        "target_info": []
    }
    fake_result["model_data"] = {"objects": []}
    fake_result ["algorithm_data"]["target_info"]=[]
    return json.dumps(fake_result, indent = 4)

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