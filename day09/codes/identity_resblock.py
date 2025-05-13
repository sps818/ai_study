import torch.nn as nn
import torch


class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 处理维度不匹配的情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)  # 跳跃连接
        return nn.ReLU()(out)


if __name__ == "__main__":
    # 假设输入通道为3，输出通道为16，步幅为1
    block = BasicResBlock(in_channels=3, out_channels=16, stride=1)
    # 构造一个batch size为4，3通道，32x32的输入
    x = torch.randn(4, 3, 32, 32)
    y = block(x)
    print("输出shape:", y.shape)
