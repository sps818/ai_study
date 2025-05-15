import cv2
import os
# 指定要处理的文件路径
datpath = 'data/'
# 获取算法的XML文件
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("haarcascade_frontalface_alt2.xml")
# 遍历所有的图片
for img in os.listdir(datpath):
    # 读取图像
    frame = cv2.imread(datpath + img)
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测结果
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 遍历检测结果
    for(x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 将检测结果写入图像中
    cv2.imwrite('result/' + img, frame)
