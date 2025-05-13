import torch.nn as nn
import torch


class BottleneckResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = nn.ReLU()(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(residual)
        return nn.ReLU()(out)


if __name__ == "__main__":
    # 假设输入通道为64，输出通道为256，步幅为1
    block = BottleneckResBlock(in_channels=64, out_channels=256, stride=1)
    # 构造一个batch size为4，64通道，32x32的输入
    x = torch.randn(4, 64, 32, 32)
    y = block(x)
    print("输出shape:", y.shape)
