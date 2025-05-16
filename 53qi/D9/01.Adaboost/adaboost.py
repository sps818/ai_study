import cv2

img = cv2.imread('1.jpg')  # 读取一张图片

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转化成灰度

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load('haarcascade_frontalface_alt2.xml')  # 一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar人脸特征分类器'''
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (50, 255, 50), 5)
    # windows
    cv2.imshow('img', img)
    cv2.waitKey()
    # linux
    # cv2.imwrite('img.jpg', img)
