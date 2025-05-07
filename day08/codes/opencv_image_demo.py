import cv2

# 读取图像
img = cv2.imread("./day08/codes/cat.jpg")

# 显示图像
cv2.imshow(winname="cat", mat=img)

# 按键等待
cv2.waitKey(delay=0)

# 释放资源
cv2.destroyAllWindows()
