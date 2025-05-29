from ultralytics import YOLO
import cv2
import os


# 1. 加载模型
print("Loading model...")
model = YOLO("yolo11n.pt")
print("Model loaded successfully.")

# 2. 打开视频
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
image_path = os.path.join(current_file_dir,
                          "segdata", "images", "Abyssinian_1.jpg")

# 识别
results = model(image_path)
cv2.imshow(winname="img", mat=results[0].plot())

# 释放资源
cv2.waitKey(0)
cv2.destroyAllWindows()
