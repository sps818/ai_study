import torch
from torch import nn
from torch.nn.functional import log_softmax
import copy
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    """
        根据原始论文，生成固定的位置编码（不可学习的死码）
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Embedding(nn.Module):
    """
        向量化层
    """

    def __init__(self, d_model, dict_len, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=dict_len,
                                  embedding_dim=d_model,
                                  padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * self.d_model ** 0.5


class FeedForward(nn.Module):
    """
        前馈网络层
            - 抽取注意力特征，再对特征做后处理
            - 先升
            - 再降
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.up_proj = nn.Linear(in_features=d_model, out_features=d_ff)
        self.down_proj = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class MultiHeadedAttention(nn.Module):
    """
        多头注意力
    """

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # 每一头占的维度 64
        self.d_k = d_model // h
        self.h = h
        # 参数层（Q, K, V, O）
        self.qkv = clones(module=nn.Linear(in_features=d_model, out_features=d_model), N=3)
        self.out = nn.Linear(in_features=d_model, out_features=d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.qkv, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.out(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.to(device=device)
        scores = scores.masked_fill(mask == False, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer_cons = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer_cons[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer_cons[1](x, lambda x: self.cross_attn(x, m, m, src_mask))
        return self.sublayer_cons[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_cons = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer_cons[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer_cons[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 在特征维度取均值
        mean = x.mean(-1, keepdim=True)
        # 在特征维度取标准差
        std = x.std(-1, keepdim=True)
        # 做标准化
        x = (x - mean) / (std + self.eps)
        return self.weight * x + self.bias


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, dict_len):
        super(Generator, self).__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=dict_len)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def get_model(tokenizer,
              N=6,
              d_model=512,
              d_ff=2048,
              h=8,
              dropout=0.1):
    """
        构建transformer模型
        src_dict_len: 源语言词典大小
        tgt_dict_len: 目标语言词典大小
        N: 编码解码层数
        d_model: 模型维度
        d_ff: 前向传播层维度
        h: 多头注意力的头数
        dropout: 随机失活概率
    """
    # 深拷贝函数
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed=nn.Sequential(Embedding(d_model=d_model,
                                          dict_len=tokenizer.src_dict_len,
                                          padding_idx=tokenizer.src_token2idx.get("<PAD>")),
                                position),
        tgt_embed=nn.Sequential(Embedding(d_model=d_model,
                                          dict_len=tokenizer.tgt_dict_len,
                                          padding_idx=tokenizer.tgt_token2idx.get("<PAD>")),
                                position),
        generator=Generator(d_model=d_model,
                            dict_len=tokenizer.tgt_dict_len)
    )

    return model


def inference_test(tokenizer):
    model = get_model(tokenizer=tokenizer, N=2)
    model.to(device=device)
    model.eval()
    print(model)
    with torch.no_grad():
        # [batch_size, seq_len]  [1, 10]
        src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device=device)
        src_mask = torch.ones(1, 1, 10)
        memory = model.encode(src, src_mask)
        # [batch_size, seq_len]  [1, 1]
        ys = torch.zeros(1, 1).type_as(src).to(device=device)
        for _ in range(9):
            out = model.decode(
                memory=memory,
                src_mask=src_mask,
                tgt=ys,
                tgt_mask=subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
    # 生成结果
    print("结果：", ys)


if __name__ == "__main__":
    from tokenizer import get_tokenizer
    tkz = get_tokenizer()
    inference_test(tokenizer=tkz)
