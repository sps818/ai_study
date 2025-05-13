import numpy as np
#import matplotlib.pyplot as plt
"""
save_checkpoint: 用于保存训练过程中模型的检查点的函数，以便在需要时恢复模型状态或进行推理。
Model: MindSpore中用于管理模型的类，包括训练和评估。
LossMonitor: MindSpore中用于监控损失的类，可用于实时监控训练过程中的损失。
TimeMonitor: MindSpore中用于监控训练时间的类，可用于实时监控训练过程中的时间。
create_scheduler: 用于创建学习率调度器的函数，用于动态调整学习率。
create_optimizer: 用于创建优化器的函数，用于调整模型参数以最小化损失函数。
create_loss: 用于创建损失函数的函数，用于衡量模型输出与目标之间的差异。
create_model: 用于创建模型的函数，可以是预训练模型或自定义模型。
create_dataset: 用于创建数据集的函数，可能包括加载数据和数据预处理。
create_transforms: 用于创建数据转换操作的函数，例如图像增强或标准化。
create_loader: 用于创建数据加载器的函数，用于加载数据集并生成可迭代的数据批次。
"""

"""
MindCV 是一个基于 MindSpore 框架的计算机视觉库，旨在提供一套易于使用的工具，使研究人员和开发者能够方便地训练、评估和部署各种视觉任务的模型。MindSpore 是由华为开发的一种开源深度学习框架，支持多种设备、多种形式的灵活高效计算。

在你提供的代码中，涉及到多个 MindCV 和 MindSpore 的模块，每个模块都有其独特的功能和作用。下面是对每个导入的模块的功能和用途的详细介绍，以及一些实际的应用示例：

1. create_dataset
这个函数用于创建数据集，它可以处理数据的加载、预处理和准备工作，以便用于训练或测试模型。用户可以指定数据集的类型（例如，ImageNet、CIFAR等），以及是否进行训练或测试。
2. create_transforms
该函数用于创建数据转换流程。在计算机视觉中，常见的数据预处理包括裁剪、旋转、缩放、归一化等。这些转换有助于模型训练时的数据增强和正则化。
3. create_loader
此函数用于创建数据加载器，它将处理的数据集和转换应用到批量加载中，适用于批量训练。
4. create_model
该函数用于创建模型。用户可以选择不同的模型架构（如ResNet, VGG等），并对其进行配置，以满足特定任务的需求。

5. create_loss
此函数用于创建损失函数，这是训练神经网络时衡量预测误差的一个关键部分。常见的损失函数包括交叉熵损失、均方误差损失等。
6. create_optimizer
此函数用于创建优化器，优化器负责调整模型参数以最小化损失函数。常见的优化器包括SGD、Adam等。
7. create_scheduler
此函数用于创建学习率调度器。学习率调度器可以在训练过程中调整学习率，帮助模型更好地收敛。
8. Model, LossMonitor, TimeMonitor
Model 类在 MindSpore 中封装了模型的训练、测试等过程。LossMonitor 和 TimeMonitor 是用于训练过程中监控损失和时间的工具。
9. save_checkpoint
该函数用于保存训练好的模型参数，以便将来可以重新加载模型进行测试或进一步训练。



"""
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.models import create_model

from mindcv.loss import create_loss

from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindspore import Model, LossMonitor, TimeMonitor
from mindspore.train.serialization import save_checkpoint

num_workers = 1

# path of dataset
data_dir = "pets_datasets/datasets"
network = create_model(model_name='resnet50', num_classes=21, pretrained=True)


# load datset
dataset_train = create_dataset(root=data_dir, split='train', num_parallel_workers=num_workers)
dataset_val = create_dataset(root=data_dir, split='val', num_parallel_workers=num_workers)

# Define and acquire data processing and augment operations
trans_train = create_transforms(dataset_name='ImageNet', is_training=True)
trans_val = create_transforms(dataset_name='ImageNet',is_training=False)

"""
loader_train = create_loader(
    dataset=dataset_train,  # 训练数据集
    batch_size=8,  # 每个批次的样本数
    is_training=True,  # 是否用于训练
    num_classes=21,  # 数据集中的类别数
    transform=trans_train,  # 数据变换操作
    num_parallel_workers=num_workers,  # 用于数据加载的并行工作者数
)

loader_val = create_loader(
    dataset=dataset_val,  # 验证数据集
    batch_size=2,  # 每个批次的样本数
    is_training=False,  # 是否用于训练
    num_classes=21,  # 数据集中的类别数
    transform=trans_val,  # 数据变换操作
    num_parallel_workers=num_workers,  # 用于数据加载的并行工作者数
)

"""
loader_train = create_loader(
    dataset=dataset_train,
    batch_size=8,
    is_training=True,
    num_classes=21,
    transform=trans_train,
    num_parallel_workers=num_workers,
)
loader_val = create_loader(
    dataset=dataset_val,
    batch_size=2,
    is_training=True,
    num_classes=21,
    transform=trans_val,
    num_parallel_workers=num_workers,)

"""
使用 create_optimizer 函数创建了一个优化器 opt，使用 Adam 优化器，学习率为 1e-4，作用于网络 network 的可训练参数上。
使用 create_loss 函数创建了一个损失函数 loss，使用交叉熵损失函数。
使用 Model 类创建了一个模型 model，传入网络、损失函数、优化器和度量（accuracy）。
调用模型的 train 方法进行训练，训练 10 个 epochs，在 loader_train 上训练数据，同时使用 LossMonitor 和 TimeMonitor 监控损失和训练时间，关闭 dataset_sink_mode。
使用 save_checkpoint 函数保存模型的参数到指定路径，以便在需要时恢复模型状态或进行推理。
使用 eval 方法评估模型在验证数据集 loader_val 上的性能，并将结果打印出来。
"""
# Define optimizer and loss function
opt = create_optimizer(network.trainable_params(), opt='adam', lr=1e-4)
loss = create_loss(name='CE')

# Instantiated model
model = Model(network, loss_fn=loss, optimizer=opt, metrics={'accuracy'})
model.train(1, loader_train, callbacks=[LossMonitor(5), TimeMonitor(5)], dataset_sink_mode=False)

# Save the checkpoint
save_checkpoint(network, "./resnet50_pets_best.ckpt")

# 模型评估 ： 训练结束后，我们评估模型在验证集上的准确性
res = model.eval(loader_val)
print("模型评估准确率:\n",res)



