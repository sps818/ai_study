from ultralytics import YOLO
import cv2


# 1. 加载模型
print("Loading model...")
model = YOLO("yolo11n.pt")
print("Model loaded successfully.")

# 2. 打开视频
print("Opening video...")
cap = cv2.VideoCapture("video.mp4")
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
