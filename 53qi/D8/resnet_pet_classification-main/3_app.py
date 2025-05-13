# 这段代码导入了 Gradio 库（一个用于构建交互式界面的库）、NumPy 库以及之前定义的 demo_main 模块。
# Gradio 库可以构建交互式界面，而 NumPy 是 Python 中用于科学计算的重要库。
# demo_main 模块应该包含了之前定义的函数和模型，用于图像预处理和模型推理。
import gradio as gr
import numpy as np
from demo_main import *

# 2.gradio案例：yolov8 图像分类
# 输入：图片  输出：类别标签

# 这段代码定义了一个名为 predict 的函数，该函数接受一个图像数组作为输入，并返回预测的宠物类别标签。
# 然后，定义了标题、描述和示例列表，用于构建 Gradio 的界面。

# def predict(img_ndarray):
#     """
#     预测图像中的宠物类别。

#     Parameters:
#         img_ndarray (numpy.ndarray): 输入图像的数组表示。

#     Returns:
#         str: 预测的宠物类别标签。

#     """
#     # 预处理图像
#     img = data_preprocess(img_ndarray)
#     # 使用模型进行预测
#     res = network(Tensor(img.reshape((1, 3, 224, 224)), mindspore.float32)).asnumpy()
#     # 获取预测结果对应的类别标签
#     predict_label = label_list[res[0].argmax()]
#     return predict_label

# # 定义 Gradio 应用的标题、描述和示例列表
# title = "基于resnet50的宠物分类系统"
# desc = "这是一个基于华为mindspore框架，用resnet卷积神经网络，做一个宠物分类系统 "
# examples = ['imgs/cat_0079.jpg','imgs/diao_0018.jpg',"imgs/goldfish_0089.jpg","imgs/house_0045.jpg"]

def predict(img_ndarray):
    # preprocess the image
    img = data_preprocess(img_ndarray)
    # predict model
    res = network(Tensor(img.reshape((1, 3, 224, 224)), mindspore.float32)).asnumpy()

    predict_label = label_list[res[0].argmax()]
    
    return predict_label

title = "基于resnet50的宠物分类系统"
desc = "这是一个基于华为mindspore框架，用resnet卷积神经网络，做一个宠物分类系统 "
examples = ['imgs/cat_0079.jpg','imgs/diao_0018.jpg',"imgs/goldfish_0089.jpg","imgs/house_0045.jpg"]

#创建了一个 Gradio 接口，并使用之前定义的 predict 函数作为预测函数。
#接口定义了一个图像输入和一个标签输出，用于显示预测的类别。它还定义了标题、描述和示例列表。
# 这段代码将启动一个 Gradio 应用，该应用提供了一个界面，用户可以上传图像，并查看模型对图像的预测结果。
"""
demo = gr.Interface(
    fn=predict,  # 预测函数
    inputs=gr.inputs.Image(),  # 图像输入
    outputs=gr.outputs.Label(label="resnet50预测的类别为:", num_top_classes=1, color="#7FFFAA"),  # 标签输出
    examples=examples,  # 示例列表
    title=title,  # 标题
    description=desc,  # 描述
)
demo.launch()  # 启动 Gradio 应用

"""
demo = gr.Interface(fn = predict,inputs = gr.Image(), outputs = gr.Label(label="resnet50预测的类别为:",num_top_classes=1,color="#7FFFAA"), 
                    examples = examples,title=title,description=desc,)
demo.launch()#share=True