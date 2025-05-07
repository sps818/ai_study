# 引入 OpenCV
import cv2

# 1. 建立连接，0代表电脑上插的第0个摄像头
cap = cv2.VideoCapture(0)

# 查看是否插了摄像头
print(f'摄像头是否插了：{cap.isOpened()}')

# 2. 增删改查
while True:
    # 读取一帧
    status, frame = cap.read()
    # 读取失败
    if not status:
        print('未能读取到视频帧！')
        break
    # 读取成功
    ###### 添加处理代码 开始########

    ###### 添加处理代码 结束########

    # 图像显示
    cv2.imshow(winname="demo", mat=frame)
    # 暂留一帧（按ESC键退出， ESC的ASCII码为27）； 1000 // 24 代表每秒24帧，1000 // 24 代表每帧的暂留时间
    if cv2.waitKey(delay=1000 // 24) == 27:
        break

# 3. 释放资源
cap.release()
cv2.destroyAllWindows()
