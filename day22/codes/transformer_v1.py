import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import log_softmax, pad
import math
import time
import copy
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class EncoderDecoder(nn.Module):
    """
    定义一个公共的 Encoder-Decoder 架构的框架
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        初始化方法：
            - encoder: 编码器（对象）
            - decoder: 解码器（对象）
            - src_embed: 输入预处理（对象）
            - tgt_embed: 输出侧的输入预处理（对象）
            - generator: 生成处理（对象）
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        输入并处理带掩码的输入和输出序列
        """
        # 1，通过 encoder 获取中间表达
        memory = self.encode(src, src_mask)
        # 2，通过 decoder 获取最终结果
        result = self.decode(memory, src_mask, tgt, tgt_mask)

        return result

    def encode(self, src, src_mask):
        """
        编码器处理过程
        """
        # 1，把输入的 id序列 变为向量并且加入位置编码
        src_embed_pos = self.src_embed(src)
        # 2，通过 encoder 获取中间表达
        memory = self.encoder(src_embed_pos, src_mask)
        return memory

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码过程
        """
        # 1，把 已经生成了的上文 id序列 变为 向量，再加上位置编码
        tgt_embed_pos = self.tgt_embed(tgt)
        # 2，通过 decoder 进行加码
        result = self.decoder(tgt_embed_pos, memory, src_mask, tgt_mask)
        return result


class Generator(nn.Module):
    """
    把向量维度转换为词表长度，输出每个词的概率
    """

    def __init__(self, d_model, dict_len):
        """
        初始化
            d_model：模型的向量维度，比如：512
            dict_len：词表长度，比如：2万
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=dict_len)

    def forward(self, x):
        """
        前向映射过程
        """
        # 1，特征映射（最后一个维度看作是特征维度）
        x = self.proj(x)
        return log_softmax(x, dim=-1)


def clones(module, N):
    """
    定义一个层的复制函数
        - nn.ModuleList
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    第一个 Encoder，其由N个encoder layer 构成~
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 克隆 N 个层
        self.layers = clones(layer, N)
        # 定义一个 norm 层
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        传入x及其mask，通过N层encoder layer 处理
        mask: pad_mask 消除 pad 的影响
        """
        # 经历 N 层处理
        for layer in self.layers:
            x = layer(x, mask)
        # 返回前，做一次 Norm 处理
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    自定义 LayerNorm 层
    """

    def __init__(self, features, eps=1e-6):
        """
        序列维度上做的
        features: 特征的维度或个数
        eps: epsilon 防止 标准差为零
        """
        super(LayerNorm, self).__init__()
        # 使用全1来初始化 类似于 weight
        # nn.Parameter 定义可学习的参数
        self.w = nn.Parameter(torch.ones(features))
        # 使用全0来初始化 类似与 bias
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        处理过程：
            - 1，减去均值
            - 2，除以标准差
            - 3，可以在一定程度上还原
        """
        # 1, 计算均值
        mean = x.mean(dim=-1, keepdim=True)
        # 2, 计算标准差
        std = x.std(dim=-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b


class SublayerConnection(nn.Module):
    """
    短接结构定义
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        # 定义一个 norm 层
        self.norm = LayerNorm(size)
        # 定义一个 dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "执行过程"
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    定义一个Encoder Layer
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_conns = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        encoder layer 的执行过程
        """
        # 1，先计算多头注意力
        x = self.sublayer_conns[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 2，再做 前馈处理
        return self.sublayer_conns[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    实现一个 Decoder
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 克隆 N 层
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        解码过程
            x: 已经生成了的上文
            memory：中间表达
            src_mask：pad_mask  标明 memory 中哪些是有效的
            tgt_mask：pad_mask + subsequent_mask
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    定义一个解码器层
        - self_attn
        - src_attn
        - feed_forward
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer_conns = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        执行解码的过程
        """
        # 1，对已经生成的前文内容进行自回归式提取特征
        x = self.sublayer_conns[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2，参考问题的中间表达，进一步生成答案
        x = self.sublayer_conns[1](
            x, lambda x: self.cross_attn(x, memory, memory, src_mask)
        )
        # 3，最后执行 前馈处理
        return self.sublayer_conns[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    未来词掩盖
        - 只能看左边，不能看右边
        - 我 | 爱 | 北京 | 天安门 | ！
        - 我 --> 我
        - 爱 -- > 我 | 爱
        - 北京 --> 我 | 爱 | 北京
        - 天安门 --> 我 | 爱 | 北京 | 天安门
        - ！--> 我 | 爱 | 北京 | 天安门 | ！
    - size：词的个数,  序列长度
    -
    """
    # 同一批次，任何一句话都是这样的规则，所以批量维度为 1 即可，计算时，自动广播
    attn_shape = (1, size, size)
    mask = torch.tril(torch.ones(attn_shape)).type(torch.bool)
    return mask


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    mask: [batch_size, 1, seq_len, seq_len]
    计算 带缩放的点乘式注意力

    """
    # 取出最后一个维度，
    # 特征维度 64
    d_k = query.size(-1)
    # [batch_size, h, seq_len, embed_dim] @ [batch_size, h, embed_dim, seq_len]
    # [batch_size, h, seq_len, seq_len] - 原始点乘积
    # 1， 计算了原始的点乘积
    # 2，做了缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #
    if mask is not None:
        # 在 mask 为 零 的位置，填上 -1e9 很大的负数
        scores = scores.masked_fill(mask=mask == False, value=-1e9)
    # 求概率，得到最终的分数：
    p_attn = scores.softmax(dim=-1)
    # 如果有 dropout, 则应用 dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    基于 PyTorch 设计自己的层：
        - 参数？
        - 逻辑？

    计算多头注意力
        - 1，分成多头
        - 2，按头计算注意力
        - 3，合并最终的结果
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        h：头数  8
        d_model: 向量的维度 512
        - 单头形式：
            - [batch_size, seq_len, embed_dim]
        - 多头形式：
            - [batch_size, h, seq_len, embed_dim // h]

        """
        super(MultiHeadedAttention, self).__init__()
        # 向量维度必须能被头数整除
        if d_model % h:
            raise ValueError("向量维度 d_model 必须能被 头数 h 整除！")
        # We assume d_v always equals d_k
        # 每一头的向量维度
        self.d_k = d_model // h
        self.h = h
        # Q, K, V
        # 核心参数
        self.qkv_matrices = clones(
            nn.Linear(in_features=d_model, out_features=d_model, bias=False), 3
        )
        # 多头特征合并之后的后处理
        self.out = nn.Linear(in_features=d_model, out_features=d_model, bias=True)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        正向传播过程
        """
        if mask is not None:
            # Same mask applied to all h heads.
            # [batch_size, seq_len, seq_len]
            # [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)
        # 取出批量 大小
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # [batch_size, h, seq_len, embed_dim // h]
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.qkv_matrices, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x: [batch_size, h, seq_len, embed_dim // h]
        # self.attn: [batch_size, h, seq_len, seq_len]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # x: [batch_size, h, seq_len, embed_dim // h] --> x: [batch_size, seq_len, h, embed_dim // h]
        # --> x: [batch_size, seq_len, embed_dim]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 直接激活垃圾回收，立刻释放这些变量所占的空间
        del query
        del key
        del value
        # 合并之后，再做一次处理
        # [batch_size, seq_len, embed_dim] --> [batch_size, seq_len, embed_dim]
        return self.out(x)


class PositionwiseFeedForward(nn.Module):
    """
    Feed Forward
    MLP: Multi-Layer Perceptron
    Linear
    - 功能：
        - 把多头注意力层抽取的特征做一次后处理
        - 把维度先升高，再下降
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model：原来的向量维度 512
        d_ff：中间高维度 2048
        """
        super(PositionwiseFeedForward, self).__init__()
        self.up_proj = nn.Linear(in_features=d_model, out_features=d_ff)
        self.down_proj = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class Embedding(nn.Module):
    """
    自定义向量化层
    """

    def __init__(self, dict_len, d_model):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings=dict_len, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.embed(x)
        x *= self.d_model**0.5
        return x


class PositionalEncoding(nn.Module):
    """
    位置编码
        - 给每个位置生成一个固定的向量
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # [max_len, embed_dim]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, embed_dim]
        pe = pe.unsqueeze(0)

        # 注册缓冲区变量，model.state_dict()
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        # self.pe: [1, max_len, embed_dim]
        # pe: [1, seq_len, embed_dim]

        # 根据实际序列长度取出 位置编码
        pe = self.pe[:, : x.size(1), :].requires_grad_(False)
        # 加上位置编码
        x += pe
        # dropout 处理
        x = self.dropout(x)
        return x


# 构建模型
def make_model(
    src_dict_len, tgt_dict_len, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    """
    构建模型：
        - src_dict_len: 输入侧字典的长度
        - tgt_dict_len：输出侧字典的长度
        - N：encoder 和 decoder 的重复次数
        - d_model：模型内部，特征向量的维度
        - d_ff：FF层中间的特征维度（先升维，再降维）
        - h：注意力头数
        - dropout：随机失活的概率
    """
    c = copy.deepcopy
    # 实例化了一个注意力层
    attn = MultiHeadedAttention(h, d_model)
    # 实例化一个前馈网络层
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化一个位置编码对象
    position = PositionalEncoding(d_model, dropout)

    # 实例化模型
    model = EncoderDecoder(
        encoder=Encoder(layer=EncoderLayer(d_model, c(attn), c(ff), dropout), N=N),
        decoder=Decoder(
            layer=DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N=N
        ),
        src_embed=nn.Sequential(Embedding(src_dict_len, d_model), position),
        tgt_embed=nn.Sequential(Embedding(tgt_dict_len, d_model), position),
        generator=Generator(d_model=d_model, dict_len=tgt_dict_len),
    )

    return model


if __name__ == "__main__":

    # 实例化一个model
    model = make_model(
        src_dict_len=8000,
        tgt_dict_len=15000,
        N=2,
        d_model=512,
        d_ff=2048,
        h=8,
        dropout=0.1,
    )

    model.eval()

    # 分词之后的 id 序列
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    print(src.shape)
    src_mask = torch.ones(1, 1, 10)
    memory = model.encode(src, src_mask)
    print(memory.shape)

    #  SOS 启动符号
    ys = torch.zeros(1, 1).type_as(src)

    # 训练生成
    for i in range(9):
        print(ys.shape)
        print(ys)
        ys_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        print(ys_mask.shape)
        out = model.decode(
            memory, src_mask, ys, ys_mask
        )
        print(out.shape)
        last_step = out[:, -1]
        print(last_step.shape)
        prob = model.generator(last_step)
        print(prob.shape)
        _, next_word = torch.max(prob, dim=1)
        print(next_word)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
