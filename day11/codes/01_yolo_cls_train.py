# 解决OMP问题
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 1. 引入 YOLO 
from ultralytics import YOLO

# 2. 加载/从零构建模型
model = YOLO("yolo11n-cls.yaml") 

# 3. 训练模型
results = model.train(data="gesture", epochs=10, imgsz=128, batch=8)