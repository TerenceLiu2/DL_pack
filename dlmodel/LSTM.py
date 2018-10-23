# -*- coding: utf-8 -*-

"""
@Author  : Teren
"""
import torch
import torch.nn as nn
from .BasicModule import BasicModule


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]  # torch.Tensor.topk()的输出有两项，后一项为索引
    return x.gather(dim, index)


class LSTM(BasicModule):
    def __init__(self, config, vectors):
        super(LSTM, self).__init__()
        self.config = config
        self.kmax_pooling = 50


        # LSTM
        self.embeds = nn.Embedding(config.kwargs['word_num'], config.kwargs['w2v_dim'])
        self.embeds.weight.data.copy_(vectors)
        self.bilstm = nn.LSTM(
            input_size=config.kwargs['w2v_dim'],
            hidden_size=config.kwargs['hidden_size'],
            num_layers=config.kwargs['num_layers'],
            dropout=0.5,
            bidirectional=True)

        # self.fc = nn.Linear(args.hidden_dim * 2 * 2, args.label_size)
        # 两层全连接层，中间添加批标准化层
        # 全连接层隐藏元个数需要再做修改
        self.fc = nn.Sequential(
            nn.Linear(config.kwargs['hidden_size']*2*self.kmax_pooling, 500),
            nn.Dropout(),
            nn.BatchNorm1d(500),
            nn.ReLU(inplace=True),
            nn.Linear(500, config.kwargs['label_num']),
            nn.Sigmoid()
        )

    def get_optimizer(self):
        # model 包含了embedding的参数优化，需要提出出来。单独优化。
        embed_params = list(map(id,list(self.embeds.parameters())))
        base_params = filter(lambda p: id(p) not in embed_params, self.parameters())
        self.optimizer = torch.optim.Adam([
            {'params': self.embeds.parameters(), 'lr': 2e-4},
            {'params': base_params, 'lr': self.config.kwargs['learning_rate']}
        ])
        return self.optimizer

    # 对LSTM所有隐含层的输出做kmax pooling
    def forward(self, text):
        embed = self.embeds(text)  # seq*batch*emb

        out = self.bilstm(embed)[0].permute(1, 2, 0)  # batch * hidden * seq

        pooling = kmax_pooling(out, 2, self.kmax_pooling)  # batch * hidden * kmax
        # word+article
        flatten = pooling.view(pooling.size(0), -1)

        out = self.fc(flatten)

        return out
