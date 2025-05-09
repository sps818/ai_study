import os
import torch
from tokenizer import get_tokenizer
from dataloader import get_dataloader
from model import get_model
from torch.optim.lr_scheduler import LambdaLR
from utils import rate
from utils import LabelSmoothing
from utils import SimpleLossCompute
from utils import run_epoch
from utils import greedy_decode
from utils import get_real_output


class Translation(object):
    def __init__(self, file="data.txt"):
        self.file = file
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = get_tokenizer(file=file)
        self.model = self._get_model(tokenizer=self.tokenizer, N=2)
        self.epochs = 20
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=0.5,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.lr_scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: rate(
                step, model_size=self.model.src_embed[0].d_model, factor=1.0, warmup=400
            ),
        )
        self.criterion = LabelSmoothing(size=self.tokenizer.tgt_dict_len,
                                        padding_idx=self.tokenizer.tgt_token2idx.get("<PAD>"),
                                        smoothing=0.1)
        self.loss_compute = SimpleLossCompute(self.model.generator, self.criterion)

    def _get_model(self, tokenizer, N):
        model = get_model(tokenizer=tokenizer, N=N)
        if os.path.exists(path="./.cache/model.pt"):
            model.load_state_dict(state_dict=torch.load(f="./.cache/model.pt", weights_only=True))
            print("加载预训练模型...")

        if os.path.exists(path="./.cache/model.pt"):
            model.load_state_dict(state_dict=torch.load(f="./.cache/model.pt", weights_only=True))
            print("加载本地模型成功")
        model.to(device=self.device)
        return model

    def train(self):
        """
            训练过程
        """
        # 训练集加载器
        train_dataloader = get_dataloader(tokenizer=self.tokenizer,
                                          file=self.file,
                                          part="train",
                                          batch_size=96)

        for epoch in range(self.epochs):
            print(f"######################## Epoch: {epoch + 1} ########################")
            self.model.train()
            loss, state = run_epoch(data_iter=train_dataloader,
                                    model=self.model,
                                    loss_compute=self.loss_compute,
                                    optimizer=self.optimizer,
                                    scheduler=self.lr_scheduler,
                                    mode="train",
                                    device=self.device)
            loss = loss.item()

            # 保存模型
            if not os.path.exists(path="./.cache"):
                os.mkdir(path="./.cache")
            torch.save(obj=self.model.state_dict(), f="./.cache/model.pt")

            # 测试一下
            self.infer(sentence="Am I wrong?")
            self.infer(sentence="I went to school by bike!")
            self.infer(sentence="Are you OK?")

            # 提前退出
            if loss < 0.5:
                print(f"训练提前结束, 当前损失为：{loss}")
                break

    def infer(self, sentence="Am I wrong?"):
        """
            预测过程
        """
        print("原文：", sentence)
        sentence = self.tokenizer.split_src(sentence=sentence)
        print("分词：", sentence)
        sentence = self.tokenizer.encode_src(src_sentence=sentence, src_sentence_len=len(sentence))
        print("编码：", sentence)
        src = torch.LongTensor([sentence]).to(device=self.device)
        print("张量：", src)
        # src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        max_len = src.size(1)
        src_mask = torch.ones(1, 1, max_len)
        # 模型推理
        self.model.eval()
        with torch.no_grad():
            y_pred = greedy_decode(self.model,
                                   src,
                                   src_mask,
                                   max_len=self.tokenizer.tgt_max_len,
                                   start_symbol=self.tokenizer.tgt_token2idx.get("<SOS>"))
        # 去除 启动信号 <SOS>
        y_pred = y_pred[:, 1:]
        raw_results, final_results = get_real_output(y_pred.cpu(), self.tokenizer)
        print("原始预测：", raw_results[0])
        print("最终预测：", final_results[0])


if __name__ == '__main__':
    translation = Translation()
    translation.train()
    translation.infer(sentence="This is my car!")
