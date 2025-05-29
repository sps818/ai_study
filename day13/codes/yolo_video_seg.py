from ultralytics import YOLO
import cv2
import os


# 1. 加载模型
print("Loading model...")
model = YOLO("yolo11n-seg.pt")
print("Model loaded successfully.")

# 2. 打开视频
print("Opening video...")

# 获取当前代码文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前代码文件的目录
current_file_dir = os.path.dirname(current_file_path)
# 获取视频文件的绝对路径
video_path = os.path.join(current_file_dir, "video.mp4")


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
print("Video opened successfully.")

while cap.isOpened():
    # 读取一帧图像
    status, frame = cap.read()
    if status:
        results = model(frame)
        img = results[0].plot()
        cv2.imshow(winname="img", mat=img)

        if cv2.waitKey(delay=1000 // 24) == 27:
            break
    else:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
