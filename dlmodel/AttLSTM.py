# -*- coding: utf-8 -*-

"""

@Author  : Teren

"""

import torch
import torch.nn as nn
from .BasicModule import BasicModule


class AttLSTM(BasicModule):
    def __init__(self, config, vectors):
        super(AttLSTM, self).__init__()
        self.config = config

        # LSTM
        self.embeds = nn.Embedding(config.kwargs['word_num'], config.kwargs['w2v_dim'])
        self.embeds.weight.data.copy_(vectors)
        self.bilstm = nn.LSTM(
            input_size=config.kwargs['w2v_dim'],
            hidden_size=config.kwargs['hidden_size'],
            num_layers=config.kwargs['num_layers'],
            dropout=0.5,
            bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(config.kwargs['hidden_size']*2, 500),
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

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.unsqueeze(2)
        # (batch, seq, hidden) * (batch, hidden, 1) = (batch, seq, 1)
        weights = torch.bmm(rnn_out.permute(0, 2, 1), merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        # (batch, hidden, seq) * (batch, seq, 1) = (batch, hidden, 1)
        return torch.bmm(rnn_out, weights).squeeze(2)

    def forward(self, text):
        embed = self.embeds(text)  # seq * batch * emb

        out,(h_n,c_n)= self.bilstm(embed)
        out = out.permute(1, 2, 0)  # batch * hidden * seq
        # h_n, c_n = hidden
        att_out = self.attention(out, c_n)

        logits = self.fc(att_out)
        return logits
