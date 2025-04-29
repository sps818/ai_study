import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os


class Seq2SeqDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, tokenizer, file="./data.txt", part="train"):
        self.tokenizer = tokenizer
        self.file = file
        self.part = part
        self.data = None
        self._load_data()

    def _load_data(self):
        if os.path.exists(path=fr"./.cache/{self.part}.bin"):
            self.data = torch.load(f=fr"./.cache/{self.part}.bin", weights_only=True)
            print("加载本地数据集成功")
            return

        data = []
        with open(file=self.file, mode="r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                if line:
                    src_sentence, tgt_sentence = line.strip().split("\t")
                    src_sentence = self.tokenizer.split_src(src_sentence)
                    tgt_sentence = self.tokenizer.split_tgt(tgt_sentence)
                    data.append([src_sentence, tgt_sentence])
        train_data, test_data = train_test_split(data,
                                                 test_size=0.2,
                                                 random_state=0)
        if self.part == "train":
            self.data = train_data
        else:
            self.data = test_data

        # 保存数据
        if not os.path.exists(".cache"):
            os.mkdir(path=".cache")
        torch.save(obj=self.data, f=fr"./.cache/{self.part}.bin")

    def __getitem__(self, idx):
        """
        返回一个样本
            - 列表格式
            - 内容 + 实际长度
        """
        src_sentence, tgt_sentence = self.data[idx]

        return (
            src_sentence,
            len(src_sentence),
            tgt_sentence,
            len(tgt_sentence)
        )

    def __len__(self):
        return len(self.data)


def collate_fn(batch, tokenizer):
    # 合并整个批量的每一部分
    src_sentences, src_sentence_lens, tgt_sentences, tgt_sentence_lens = zip(
        *batch
    )

    # 转索引【按本批量最大长度来填充】
    src_sentence_len = max(src_sentence_lens)
    src_idxes = []
    for src_sentence in src_sentences:
        src_idxes.append(tokenizer.encode_src(src_sentence, src_sentence_len))

    # 转索引【按本批量最大长度来填充】
    tgt_sentence_len = max(tgt_sentence_lens)
    tgt_idxes = []
    for tgt_sentence in tgt_sentences:
        tgt_idxes.append(
            tokenizer.encode_tgt(tgt_sentence, tgt_sentence_len)
        )
    # 转张量 [batch_size, seq_len]  src
    src_idxes = torch.LongTensor(src_idxes)
    # src_mask [batch_size, 1, seq_len]
    src_mask = (src_idxes != tokenizer.src_token2idx.get("<PAD>")).unsqueeze(-2)
    # tgt [batch_size, seq_len]
    tgt_idxes = torch.LongTensor(tgt_idxes)
    # tgt [batch_size, seq_len - 1] 去掉最后的 EOS
    tgt_idxes_in = tgt_idxes[:, :-1]
    # tgt_y [batch_size, seq_len - 1] 去掉开头 的 SOS
    tgt_idxes_out = tgt_idxes[:, 1:]
    # tgt_mask [batch_size, seq_len-1, seq_len-1]
    tgt_mask = tokenizer.make_std_mask(tgt_idxes_in,
                                          pad=tokenizer.tgt_token2idx.get("<PAD>"))
    # 记录生成的有效字符
    ntokens = (tgt_idxes_out != tokenizer.tgt_token2idx.get("<PAD>")).data.sum()
    # src, src_mask, tgt, tgt_mask, tgt_y, ntokens
    return src_idxes, src_mask, tgt_idxes_in, tgt_mask, tgt_idxes_out, ntokens


def get_dataloader(tokenizer,
                   file=r"./data.txt",
                   part="train",
                   batch_size=128):
    dataset = Seq2SeqDataset(file=file, tokenizer=tokenizer, part=part)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if part == "train" else False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    return dataloader


if __name__ == "__main__":
    from tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    dataloader = get_dataloader(tokenizer=tokenizer, part="test")
    for src, src_mask, tgt, tgt_mask, tgt_y, ntokens in dataloader:
        # [batch_size, src_max_seq_len]
        print(src.shape)
        # [batch_size, 1, src_max_seq_len]
        print(src_mask.shape)
        # [batch_size, tgt_max_seq_len-1]
        print(tgt.shape)
        # [batch_size, tgt_max_seq_len-1, tgt_max_seq_len-1]
        print(tgt_mask.shape)
        # [batch_size, tgt_max_seq_len-1]
        print(tgt_y.shape)
        # 7784
        print(ntokens)
        break
