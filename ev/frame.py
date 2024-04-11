import cv2

# 视频文件路径
video_path = './video/deepsort.avi'
capture = cv2.VideoCapture(video_path)

# 输出图片的保存路径
output_path = './frames4/'

# 创建输出图片文件夹
import os
os.makedirs(output_path, exist_ok=True)

# 逐帧读取视频
frame_id = 0
while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        break

    # 将每一帧保存为图片
    frame_path = os.path.join(output_path, f'frame_{frame_id}.png')
    cv2.imwrite(frame_path, frame)

    frame_id += 1

# 释放资源
capture.release()
cv2.destroyAllWindows()
