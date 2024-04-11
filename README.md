# In-Vehicle_MOT
Enhancing Multi-Object Tracking: YOLOv5 and Advanced Tracking Algorithms

**代码整体分为两部分，目标检测和目标跟踪**

- 目标检测模型训练部分
极市平台上训练部分的代码保存在train文件夹内,运行train/src_repo/sun_detect.sh即可开始训练，训练产生的模型保存在models文件夹下
- 多目标跟踪测试部分
目标跟踪的相关代码保存在ev文件夹内，进入ev文件夹内运行`detect_track.py`即可对视频进行标注，并将结果保存到video目录下，`ji.py`是用于极市平台测试的接口文件。
运行`frame.py`可以将标注好的视频转存为连续帧的图片。

**比赛阶段用到的三个算法**

Sort:`kaiman_tracker.py`、`track_main.py`、`ssd_detector.py`、`tools.py`
DeepSort:`deep_sort`目录下
ByteTrack:`bytetrack_tracker`目录下
对于bytetrack算法所做的改进部分集中在`bytetrack_tracker/byte_tracker.py`文件中