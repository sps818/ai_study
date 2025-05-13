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
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]


def _normalize(img, mean, std):
    # This method is borrowed from:
    
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


def data_preprocess(img_ndarray):
    #img = cv2.imread(img_path, 1)
    img = img_ndarray
    img = cv2.resize(img, (256, 256))
    img = _crop_center(img, 224, 224)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    img = _normalize(img.astype(np.float32), np.asarray(mean), np.asarray(std))
    img = img.transpose(2, 0, 1)

    return img




# if __name__ == '__main__':
#     image_path = r"imgs/raccoon_0106.jpg"

#     # preprocess the image
#     img = data_preprocess(image_path)
#     # predict model
#     res = network(Tensor(img.reshape((1, 3, 224, 224)), mindspore.float32)).asnumpy()

#     predict_label = label_list[res[0].argmax()]
#     print()
#     print("预测的宠物类别为:\n"+predict_label+"\n")