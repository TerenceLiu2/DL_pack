# -*- coding: utf-8 -*-
'''

@author : 使用Conv2d的TextCNN

'''
import torch
import torch.nn as nn
from .BasicModule import BasicModule

kernel_sizes = [2,3,4,5,6,7,8]

class TextCNN(BasicModule):

    def __init__(self, config,vector=None):
        super(TextCNN, self).__init__()
        self.config = config
        self.embeds = nn.Embedding(config.kwargs['word_num'], config.kwargs['w2v_dim'])
        self.MaxPool = nn.MaxPool2d((self.config.kwargs['sentence_max_size'], 1))
        self.AvgPool = nn.AvgPool2d((self.config.kwargs['sentence_max_size'], 1)) #类不均衡也很关键


        if vector is not None:
            self.embeds.weight.data.copy_(vector)

        convs = [
            nn.Sequential(
                nn.Conv2d(1, 1, (kernel_size, config.kwargs['w2v_dim'])),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, (kernel_size, 1)),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d((self.config.kwargs['sentence_max_size'], 1)),
                # nn.AvgPool2d((self.config.kwargs['sentence_max_size'], 1))
                # AvgPool
            )
            for kernel_size in kernel_sizes
        ]

        self.convs = nn.ModuleList(convs)

        self.fc = nn.Sequential(
            nn.Linear(4 * len(kernel_sizes), 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.config.kwargs['label_num']),
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

    def forward(self, x):

        x = torch.from_numpy(x.numpy().T).long()

        batch = x.shape[0]
        x = self.embeds(x)

        # x = x.reshape(batch,self.config.kwargs['w2v_dim'],self.config.kwargs['padding_size'])

        x = x.reshape(batch, 1, self.config.kwargs['padding_size'],self.config.kwargs['w2v_dim'])

        print(x.shape)

        conv_out = [conv(x) for conv in self.convs]

        # print(conv_out[0].shape)

        max_out = [self.MaxPool(x) for x in conv_out]
        avg_out = [self.AvgPool(x) for x in conv_out]

        max_out.extend(avg_out)

        # Needs Dropout

        # capture and concatenate the features
        x = torch.cat(([x for x in max_out]), -1)
        x = x.view(batch, -1)

        # project the features to the labels
        x = self.fc(x)

        x = x.view(-1, self.config.kwargs['label_num'])

        return x
