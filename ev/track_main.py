import os
import cv2
import numpy as np
from collections import deque
from ssd_detector import CarDetectot_SSD
from Mytools import draw_box_label, box_iou, calculate_iou
from scipy.optimize import linear_sum_assignment
from kalman_tracker import Kalman_Tracker
from glob import glob
import xml.etree.ElementTree as ET
import pickle
from os import listdir, getcwd
from os.path import join
import logging
from random import randint
import time
import re

frame_count = 0         # 帧计数器
max_age = 8           # 若一个跟踪目标长时间未被检测到,则删除这个跟踪目标,该参数设置多久未跟踪到目标时删除
min_hits = 2           # 检测到同一目标多少次后进行跟踪

tracker_list = []        # 跟踪目标列表
type_list = []
# # 跟踪目标ID列表
# track_id_list = deque(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
# 使用 range 函数生成 1 到 100 的整数列表，并将其转换为字符串
track_id_list = deque(map(str, range(1, 101)))

debug = False


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    '''
    将已有的跟踪目标与新检测到的目标进行匹配
    '''
    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)  # IOU矩阵
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            IOU_mat[t, d] = calculate_iou(trk, det)

    # 匹配算法-匈牙利算法

    matched_idx = linear_sum_assignment(-IOU_mat)
    matched_idx = np.asarray(matched_idx)
    matched_idx = np.transpose(matched_idx)
    # 该函数后续版本删除,替代方案如下:https://www.cnblogs.com/clemente/p/12321745.html

    """
    sklearn API result:
    [[0 1]
    [1 0]
    [2 2]]
    """
    # 计算跟踪目标和检测目标中没有被检测出来的部分
    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # 删除IOU小于阈值的匹配点,这些可能是噪声
    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    # 匹配到的目标
    if (len(matches) == 0):         # 如果找不到匹配的
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def pipeline(bbox, cls,path = None):

    global frame_count
    global tracker_list
    global track_id_list
    global type_list
    
    frame_count += 1

    # print("frame_count:", frame_count)

    detect_boxes = bbox    # 检测到的Boxes框
    # print("detect_boxes", detect_boxes)
    # img_draw = draw_box_label(image, detect_boxes, box_color=(255, 0, 0))
    # cv2.imshow("detector image", img_draw)
    # cv2.waitKey(0)

    track_boxes = []
    if len(tracker_list) > 0:       # 已经有待跟踪的目标
        for trk in tracker_list:
            track_boxes.append(trk.box)

    # 已经提取到待追踪的目标和检测到新的目标,需进行目标匹配
    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(track_boxes, detect_boxes, iou_thrd=0.25)
    # print('Detection: ', detect_boxes)
    # print('track_boxes: ', track_boxes)
    # print('matched:', matched)
    # print('unmatched_det:', unmatched_dets)
    # print('unmatched_trks:', unmatched_trks)


    # 处理匹配成功的detections
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = detect_boxes[det_idx]           # 取出检测Box(也即测量值)
            z = np.expand_dims(z, axis=0).T     # 维度变换,便于操作
            tmp_trk = tracker_list[trk_idx]     # 取出当前跟踪目标
            tmp_trk.predict_update(z)           # 卡尔曼更新
            new_box = tmp_trk.x_state.T[0].tolist()
            new_box = [new_box[0], new_box[2], new_box[4], new_box[6]]
            track_boxes[trk_idx] = new_box
            tmp_trk.box = new_box
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # 处理未匹配的检测到的目标
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = detect_boxes[idx]
            type = cls[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Kalman_Tracker()              # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            new_box = tmp_trk.x_state.T[0].tolist()
            new_box = [new_box[0], new_box[2], new_box[4], new_box[6]]
            tmp_trk.box = new_box
            tmp_trk.id = track_id_list.popleft()
            tracker_list.append(tmp_trk)
            type_list.append(type)
            track_boxes.append(new_box)

    # 处理未匹配的跟踪目标
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            new_box = tmp_trk.x_state.T[0].tolist()
            new_box = [new_box[0], new_box[2], new_box[4], new_box[6]]
            tmp_trk.box = new_box
            track_boxes[trk_idx] = new_box

    # The list of tracks to be annotated
    good_tracker_list = []
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):     # 显示更新后的boxes
            good_tracker_list.append(trk)
            # img_draw = draw_box_label(img_draw, [trk.box], trk.id, (0, 0, 255))
    # out_detect.write(img_draw)

    # cv2.imshow("update frame", img_draw)
    # cv2.waitKey(0)

    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)
    for trk in deleted_tracks:
        track_id_list.append(trk.id)

    idx_list = [i for i, x in enumerate(tracker_list) if x.no_losses <= max_age]
    tracker_list = [x for i,x in enumerate(tracker_list) if i in idx_list]
    type_list = [x for i,x in enumerate(type_list) if i in idx_list]

    # print('Ending tracker_list: ', len(tracker_list))
    # print('Ending good tracker_list: ', len(good_tracker_list))
    
    # x, y, width, height, vechile_type, id
    x = []
    y = []
    width = []
    height = []
    vechile_type = []
    id = []
    for i, trk in enumerate(tracker_list):
        bbox = trk.box
        x.append(bbox[0])
        y.append(bbox[1])
        width.append(bbox[2] - bbox[0])
        height.append(bbox[3] - bbox[1])
        vechile_type.append(type_list[i])
        id.append(trk.id)
    return x, y, width, height, vechile_type, id

def get_bbox(image_path):
    in_file = open(image_path.replace('.xml', '.xml'), encoding="utf-8")
    tree = ET.parse(in_file)
    root = tree.getroot()
    bbox = []
    # 遍历XML树中的每个object元素
    for obj in root.iter('object'):
        # 提取bbox
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymax').text))
        b = list(b)
        bbox.append(b)
    return bbox


if __name__ == "__main__":
    car_detector = CarDetectot_SSD()    # 生成基于SSD的车辆检测器

    # images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./trainval/2850/"))
    # images_name = os.listdir(images_path)

    labels = glob('trainval/2850/*.xml')
    for i in range(len(labels)):
        # image_single_path = os.path.abspath(os.path.join(images_path, image_single_name))

        bbox = get_bbox(labels[i])
        pipeline(bbox)
        for track in tracker_list:
            print(track.id)

