# 引入 OpenCV
import cv2

# 1. 建立连接
cap = cv2.VideoCapture(0)

# 2. 增删改查
while True:
    # 读取一帧
    status, frame = cap.read()
    # 读取失败
    if not status:
        print('error')
        break
    # 读取成功
    # 添加处理代码
        
    # 图像显示
    cv2.imshow(winname="demo", mat=frame)
    # 暂留一帧（按ESC键退出）
    if cv2.waitKey(delay=1000 // 24) == 27:
        break

# 3. 释放资源
cap.release()
cv2.destroyAllWindows()

    