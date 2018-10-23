# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-8-10 下午2:58
@ide     : PyCharm  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule



def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class RCNN(BasicModule):


    def get_optimizer(self):
        # model 包含了embedding的参数优化，需要提出出来。单独优化。
        embed_params = list(map(id,list(self.embeds.parameters())))
        base_params = filter(lambda p: id(p) not in embed_params, self.parameters())
        self.optimizer = torch.optim.Adam([
            {'params': self.embeds.parameters(), 'lr': 2e-4},
            {'params': base_params, 'lr': self.config.kwargs['learning_rate']}
        ])
        return self.optimizer

    def __init__(self, config, vectors=None):
        super(RCNN, self).__init__()
        self.kmax_pooling = 2
        self.config = config

        #
        self.embeds = nn.Embedding(config.kwargs['word_num'], config.kwargs['w2v_dim'])
        self.embeds.weight.data.copy_(vectors)
        self.lstm = nn.LSTM(
            input_size=config.kwargs['w2v_dim'],
            hidden_size=config.kwargs['hidden_size'],
            num_layers=config.kwargs['num_layers'],
            dropout=0.5,
            batch_first=False,
            bidirectional=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=config.kwargs['hidden_size'] * 2 + config.kwargs['w2v_dim'], out_channels=200, kernel_size=3),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True)
        )

        # classifer
        # self.fc = nn.Linear(2 * (100 + 100), args.label_size)
        self.fc = nn.Sequential(
            nn.Linear(200*self.kmax_pooling, 1000),
            # nn.Dropout(),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, config.kwargs['label_num']),
            nn.Sigmoid()
        )

    def forward(self, text):
        embed = self.embeds(text)
        out = self.lstm(embed)[0].permute(1, 2, 0)
        out = torch.cat((out, embed.permute(1, 2, 0)), dim=1)
        conv_out = kmax_pooling(self.conv(out), 2, self.kmax_pooling) #batch * 200 * k-max
        flatten = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(flatten)
        return logits
