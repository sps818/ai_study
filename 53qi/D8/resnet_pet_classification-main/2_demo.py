"""
这段代码首先导入了需要的库，并加载了预训练的 ResNet50 模型的参数。
然后，设置了模型的 training 标志为 False，以将模型切换到推理模式。
具体步骤如下：
导入了 cv2、numpy、mindspore 库，并从 mindspore 中导入了 Tensor 类。
使用 create_model 函数创建了一个 ResNet50 模型 network，并指定了类别数为 21，并加载了预训练的参数。
定义了模型参数的路径 checkpoint_path，以及类别标签列表 label_list。
使用 load_checkpoint 函数加载了预训练模型的参数，并将参数保存在 param_dict 中。
使用 load_param_into_net 函数将加载的参数加载到创建的网络模型 network 中。
最后，调用了 set_train(False) 方法，将模型设置为推理模式，即关闭了梯度计算，因为在推理阶段不需要进行梯度更新。
"""
import cv2
import numpy as np
import mindspore
from mindspore import Tensor
from mindcv.models import create_model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

# 模型初始化
network = create_model(model_name='resnet50', num_classes=21, pretrained=True)

checkpoint_path = "models/resnet50_pets_best.ckpt"
label_list =["cat","diao","dog","duck","fox","goldfish","guinea-pig","hamster","hare","hedgehog","house","houzi","parrot","pig","raccoon","snake","songsu","wa","wugui","xiyi","yangtuo"]

# load checkpoint
param_dict = load_checkpoint(checkpoint_path)
load_param_into_net(network, param_dict)
network.set_train(False)



def _crop_center(img, cropx, cropy):
    """
    从图像的中心裁剪指定大小的区域。
    Parameters:
        img (numpy.ndarray): 输入图像，形状为 (height, width, channels)。
        cropx (int): 欲裁剪的宽度。
        cropy (int): 欲裁剪的高度。

    Returns:
        numpy.ndarray: 裁剪后的图像。
    """
    # 获取图像的尺寸
    y, x, _ = img.shape
    # 计算裁剪的起始位置，使得裁剪区域位于图像中心
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    # 返回裁剪后的图像区域
    return img[starty:starty + cropy, startx:startx + cropx, :]

"""
def _normalize(img, mean, std):

    # 对图像进行归一化。
    # Parameters:
    #     img (numpy.ndarray): 输入图像，形状为 (height, width, channels)。
    #     mean (numpy.ndarray): 均值数组，形状为 (3,)，表示图像的 RGB 通道均值。
    #     std (numpy.ndarray): 标准差数组，形状为 (3,)，表示图像的 RGB 通道标准差。
    # Returns:
    #     numpy.ndarray: 归一化后的图像。
    # 检查输入图像的数据类型是否为 uint8
    assert img.dtype != np.uint8
    # 将均值转换为浮点数并重塑为 (1, 3) 形状
    mean = np.float64(mean.reshape(1, -1))
    # 计算标准差的倒数，转换为浮点数并重塑为 (1, 3) 形状
    stdinv = 1 / np.float64(std.reshape(1, -1))
    # 将图像从 BGR 格式转换为 RGB 格式
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    # 从图像中减去均值
    cv2.subtract(img, mean, img)
    # 将图像乘以标准差的倒数
    cv2.multiply(img, stdinv, img)
    # 返回归一化后的图像
    return img
"""

def _normalize(img, mean, std):
    # This method is borrowed from:
    #   https://github.com/open-mmlab/mmcv/blob/master/mmcv/image/photometric.py
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img

# def data_preprocess(img_path):
#     """
#     对输入图像进行预处理，包括读取、缩放、裁剪和归一化等操作。

#     Parameters:
#         img_path (str): 输入图像的文件路径。

#     Returns:
#         numpy.ndarray: 预处理后的图像，形状为 (channels, height, width)。

#     """
#     # 使用 OpenCV 读取图像，参数 1 表示以 RGB 模式读取
#     img = cv2.imread(img_path, 1)
#     # 将图像调整为指定的大小 (256, 256)
#     img = cv2.resize(img, (256, 256))
#     # 从图像中心裁剪出大小为 (224, 224) 的区域
#     img = _crop_center(img, 224, 224)
#     # 定义图像的均值和标准差
#     mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
#     std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
#     # 对图像进行归一化，并将数据类型转换为 float32
#     img = _normalize(img.astype(np.float32), np.asarray(mean), np.asarray(std))
#     # 将图像的通道维度从最后一个位置移动到第一个位置
#     img = img.transpose



def data_preprocess(img_path):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (256, 256))
    img = _crop_center(img, 224, 224)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    img = _normalize(img.astype(np.float32), np.asarray(mean), np.asarray(std))
    img = img.transpose(2, 0, 1)

    return img




# 这个 if __name__ == '__main__': 块是程序的主入口点。
# 它检查当前模块是否被直接执行，而不是被导入到另一个模块中。如果是直接执行的话，将执行其中的代码。
# 这段代码首先指定了图像文件的路径，然后调用 data_preprocess 函数对图像进行预处理。
# 接着，将预处理后的图像传递给网络进行预测，并获取预测结果对应的类别标签。最后，打印出预测的宠物类别。

if __name__ == '__main__':
    image_path = r"imgs/raccoon_0106.jpg"

    # preprocess the image
    img = data_preprocess(image_path)
    # predict model
    res = network(Tensor(img.reshape((1, 3, 224, 224)), mindspore.float32)).asnumpy()

    predict_label = label_list[res[0].argmax()]
    print()
    print("预测的宠物类别为:\n"+predict_label+"\n")

# if __name__ == '__main__':
#     # 定义图像文件路径
#     image_path = r"imgs/raccoon_0106.jpg"

#     # 预处理图像
#     img = data_preprocess(image_path)

#     # 使用模型进行预测
#     res = network(Tensor(img.reshape((1, 3, 224, 224)), mindspore.float32)).asnumpy()

#     # 获取预测结果对应的类别标签
#     predict_label = label_list[res[0].argmax()]

#     # 打印预测结果
#     print()
#     print("预测的宠物类别为:\n"+predict_label+"\n")
