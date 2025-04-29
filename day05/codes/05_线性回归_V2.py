import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from typing import Tuple, Optional


def preprocess_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    数据预处理：标准化
    Args:
        X_train: 训练数据
        X_test: 测试数据
    Returns:
        标准化后的训练数据和测试数据
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_test


class LinearRegression:
    def __init__(self, input_dim: int):
        """
        初始化线性回归模型
        Args:
            input_dim: 输入特征维度
        """
        self.w = torch.randn(input_dim, 1, requires_grad=True)
        self.b = torch.randn(1, 1, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入数据
        Returns:
            预测结果
        """
        return x @ self.w + self.b

    def train(self,
              X_train: torch.Tensor,
              y_train: torch.Tensor,
              steps: int = 10001,
              learning_rate: float = 1e-2,
              print_interval: int = 1000) -> None:
        """
        训练模型
        Args:
            X_train: 训练数据
            y_train: 训练标签
            steps: 训练步数
            learning_rate: 学习率
            print_interval: 打印间隔
        """
        for step in range(steps):
            # 前向传播
            y_pred = self.forward(X_train)
            loss = ((y_train - y_pred) ** 2).mean()

            # 反向传播
            loss.backward()

            # 参数更新
            self.w.data -= learning_rate * self.w.grad
            self.b.data -= learning_rate * self.b.grad

            # 梯度清零
            self.w.grad.zero_()
            self.b.grad.zero_()

            # 打印训练信息
            if step % print_interval == 0:
                print(f'第{step+1}步的 MSE:{loss.item():.4f}')

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测
        Args:
            X_test: 测试数据
        Returns:
            预测结果
        """
        X_test = torch.tensor(data=X_test, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X_test).view(-1).numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        评估模型
        Args:
            X_test: 测试数据
            y_test: 测试标签
        Returns:
            MSE损失
        """
        y_pred = self.predict(X_test)
        return ((y_pred - y_test) ** 2).mean()

    def save_model(self, path: str) -> None:
        """
        保存模型
        Args:
            path: 保存路径
        """
        torch.save({
            'w': self.w,
            'b': self.b
        }, path)

    @staticmethod
    def load_model(path: str) -> 'LinearRegression':
        """
        加载模型
        Args:
            path: 模型路径
        Returns:
            加载的模型
        """
        checkpoint = torch.load(path)
        model = LinearRegression(checkpoint['w'].shape[0])
        model.w.data = checkpoint['w']
        model.b.data = checkpoint['b']
        return model


def main():
    # 1. 数据加载
    data = pd.read_csv(
        filepath_or_buffer="./day05/codes/boston_house_prices.csv",
        skiprows=1
    )
    data = data.to_numpy()
    X = data[:, :-1]
    y = data[:, -1]

    # 2. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # 3. 数据预处理
    X_train, X_test = preprocess_data(X_train, X_test)

    # 4. 转换为张量
    X_train = torch.tensor(data=X_train, dtype=torch.float32)
    y_train = torch.tensor(data=y_train.reshape(-1, 1), dtype=torch.float32)

    # 5. 创建模型
    model = LinearRegression(input_dim=X_train.shape[1])

    # 6. 训练模型
    model.train(X_train, y_train, steps=20001, learning_rate=1e-3)

    # 7. 评估模型
    mse = model.evaluate(X_test, y_test)
    print(f'测试集MSE: {mse:.4f}')

    # 8. 保存模型
    model.save_model('linear_regression_model.pth')

    # 9. 加载模型示例
    # loaded_model = LinearRegression.load_model('linear_regression_model.pth')


if __name__ == '__main__':
    main()
