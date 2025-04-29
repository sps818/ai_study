import jieba
from opencc import OpenCC
from tqdm import tqdm
import torch
import os

# 繁体转简体
opencc = OpenCC("t2s")


class Tokenizer(object):
    """
        定义分词器
            1. 分词 cut
            2. 字典
            3. encode：句子 --> 词 --> 变 ids
            4. decode：ids --> 词 --> 句子
    """

    def __init__(self, file="./data.txt", saved_dict="./.cache/dicts.bin"):
        """
            初始化
        """
        self.file = file
        self.saved_dict = saved_dict
        self.src_token2idx = {}
        self.src_idx2token = {}
        self.src_dict_len = None
        self.src_embed_dim = 512
        self.src_hidden_size = 512

        self.tgt_token2idx = {}
        self.tgt_idx2token = {}
        self.tgt_dict_len = None

        # 推理时，输出的最大长度
        self.tgt_max_len = 100
        self.tgt_embed_dim = 512
        self.tgt_hidden_size = 512

    def build_dict(self):
        """
        构建字典
            file: 训练数据集的文件
        """
        if os.path.exists(path=self.saved_dict):
            self.load()
            print("加载本地字典成功")
            return

        # 不需要设置启动和结束
        src_tokens = {"<UNK>", "<PAD>"}
        # 生成侧需要启动和结束 special tokens
        tgt_tokens = {"<UNK>", "<PAD>", "<SOS>", "<EOS>"}

        with open(file=self.file, mode="r", encoding="utf8") as f:
            for line in tqdm(f.readlines()):
                if line:
                    src_sentence, tgt_sentence = line.strip().split("\t")
                    src_sentence_tokens = self.split_src(src_sentence)
                    tgt_sentence_tokens = self.split_tgt(tgt_sentence)
                    src_tokens = src_tokens.union(set(src_sentence_tokens))
                    tgt_tokens = tgt_tokens.union(set(tgt_sentence_tokens))
        # 输入字典
        self.src_token2idx = {token: idx for idx, token in enumerate(src_tokens)}
        self.src_idx2token = {idx: token for token, idx in self.src_token2idx.items()}
        self.src_dict_len = len(self.src_token2idx)

        # 输出字典
        self.tgt_token2idx = {token: idx for idx, token in enumerate(tgt_tokens)}
        self.tgt_idx2token = {idx: token for token, idx in self.tgt_token2idx.items()}
        self.tgt_dict_len = len(self.tgt_token2idx)

        # 保存
        self.save()
        print("保存字典成功")

    def split_src(self, sentence):
        """
        预处理
            输入：I'm a student.
            输出：['i', 'm', 'a', 'student', '.']
        """
        # 英文变小写
        sentence = sentence.lower()
        # 把缩写拆开为两个词
        sentence = sentence.replace("'", " ")
        # 使用jieba分词
        tokens = [token for token in jieba.lcut(sentence) if token != " "]
        # 返回结果（列表形式）
        return tokens

    def split_tgt(self, sentence):
        """
        切分汉语
            输入：我爱北京天安门
            输出：['我', '爱', '北京', '天安门']
        """
        # 繁体转简体
        sentence = opencc.convert(sentence)
        # jieba 分词
        tokens = jieba.lcut(sentence)
        # 返回结果（列表形式）
        return tokens

    def encode_src(self, src_sentence, src_sentence_len):
        """
        将输入的句子，转变为指定长度的序列号
        输入：["i", "m", "a", "student"]
        输出：[5851, 4431, 6307, 1254, 2965]
        """
        # 变索引号
        src_idx = [
            self.src_token2idx.get(token, self.src_token2idx.get("<UNK>"))
            for token in src_sentence
        ]
        # 填充PAD
        src_idx = (src_idx + [self.src_token2idx.get("<PAD>")] * src_sentence_len)[:src_sentence_len]

        return src_idx

    def encode_tgt(self, tgt_sentence, tgt_sentence_len):
        """
        将输出的句子，转变为指定长度的序列号
        输入：["我", "爱", "北京", "天安门"]
        输出：[11642, 10092, 5558, 3715, 10552, 1917]
        """
        # 添加开始和结束标识符 <SOS> <EOS>
        tgt_sentence = ["<SOS>"] + tgt_sentence + ["<EOS>"]
        tgt_sentence_len += 2
        # 变 索引号
        tgt_idx = [
            self.tgt_token2idx.get(token, self.tgt_token2idx.get("<UNK>"))
            for token in tgt_sentence
        ]
        # 填充 PAD
        tgt_idx = (tgt_idx + [self.tgt_token2idx.get("<PAD>")] * tgt_sentence_len)[:tgt_sentence_len]
        return tgt_idx

    def decode_tgt(self, pred):
        """
        把预测结果转换为输出文本
        输入：[6360, 7925, 8187, 7618, 1653, 4509]
        输出：['我', '爱', '北京', '<UNK>']
        """
        results = []
        for idx in pred:
            if idx == self.tgt_token2idx.get("<EOS>"):
                break
            results.append(self.tgt_idx2token.get(idx))
        return results

    @classmethod
    def subsequent_mask(cls, size):
        """
            屏蔽未来词
                - size： seq_len
        """
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0

    @classmethod
    def make_std_mask(cls, tgt, pad):
        "Create a mask to hide padding and future words."
        """
            训练时，给 decoder 侧生成的掩码
                - 既要干掉pad
                - 又要干掉未来次
        """
        pad_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = pad_mask & Tokenizer.subsequent_mask(tgt.size(-1)).type_as(pad_mask.data)
        return tgt_mask

    def __repr__(self):
        return f"[Tokenizer]: 输入字典长度: {self.src_dict_len}, 输出字典长度: {self.tgt_dict_len}"

    def save(self):
        """
        保存字典
        """
        state_dict = {
            "src_token2idx": self.src_token2idx,
            "src_idx2token": self.src_idx2token,
            "src_dict_len": self.src_dict_len,
            "tgt_token2idx": self.tgt_token2idx,
            "tgt_idx2token": self.tgt_idx2token,
            "tgt_dict_len": self.tgt_dict_len
        }
        # 保存到文件
        if not os.path.exists(".cache"):
            os.mkdir(path=".cache")
        torch.save(obj=state_dict, f=self.saved_dict)

    def load(self):
        """
        加载字典
        """
        if os.path.exists(path=self.saved_dict):
            state_dict = torch.load(f=self.saved_dict, weights_only=True)
            self.src_token2idx = state_dict.get("src_token2idx")
            self.src_idx2token = state_dict.get("src_idx2token")
            self.src_dict_len = state_dict.get("src_dict_len")
            self.tgt_token2idx = state_dict.get("tgt_token2idx")
            self.tgt_idx2token = state_dict.get("tgt_idx2token")
            self.tgt_dict_len = state_dict.get("tgt_dict_len")


def get_tokenizer(file="./data.txt", saved_dict="./.cache/dicts.bin"):
    """
    获取分词器
    """
    # 定义分词器
    tokenizer = Tokenizer(file=file, saved_dict=saved_dict)
    tokenizer.build_dict()
    return tokenizer


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    print(tokenizer)
