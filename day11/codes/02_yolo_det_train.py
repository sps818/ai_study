# 1. 解决OMP
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

# 2. 导入 YOLO
from ultralytics import YOLO

if __name__ == '__main__':

    # 3. 加载模型
    model = YOLO("yolo11n.yaml")  # build a new model from YAML

    # 4. 训练模型
    results = model.train(data="coco8.yaml", epochs=10, imgsz=640)
