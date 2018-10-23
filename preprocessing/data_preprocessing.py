#coding=utf8

'''

@author: Teren
模型的输入数据

'''


import pandas as pd
from torch.nn import init
from tqdm import tqdm
from torchtext import data
from torchtext.vocab import Vectors
import torch
import random
import numpy as np

# config = Config(
#                 raw_train_path = '../data/train_train_seg.csv',
#                 raw_test_path = '../data/test_public_seg.csv',
#                 raw_vali_path = '../data/vali_train_seg.csv',
#                 embedding_path = '../data/300_5_w2v_word.model',
#                 padding_size = 80,
#                 batch_size = 128,
#                 aug=False,
#                 )



class GrandDataset(data.Dataset):
    name = 'Grand Dataset'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, config,test=False, **kwargs):
        '''
        :param path: 输入文件路径
        :param text_field:  输入格式应该规范为分好词后的结果，每个词之间用空格分隔
        :param label_field: 自定义标签格式
        :param config:
        :param test:
        :param kwargs:
        '''
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(csv_data['content']):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label,senti in tqdm(zip(csv_data['content'], csv_data['subject'],csv_data['sentiment_value'])):
                if config.kwargs['aug']:
                    rate = random.random()
                    if rate > 0.5:
                           pass
                    else:
                        text = self.shuffle(text)
                y_word_index = {'动力': 1, '价格': 2, '安全性': 7, '操控': 4, '空间': 10, '内饰': 8, '外观': 9, '舒适性': 5, '油耗': 3,
                                '配置': 6}
                y_wordsenti_index = {'动力,1': 17, '安全性,-1': 21, '外观,0': 7, '油耗,0': 1, '价格,0': 19, '操控,0': 28,
                                     '配置,-1': 12, '安全性,0': 22, '安全性,1': 23, '配置,1': 14, '内饰,1': 5, '空间,-1': 24,
                                     '价格,1': 20, '价格,-1': 18, '外观,-1': 6, '空间,1': 26, '外观,1': 8, '配置,0': 13, '内饰,0': 4,
                                     '动力,0': 16, '操控,-1': 27, '操控,1': 29, '油耗,-1': 0, '油耗,1': 2, '内饰,-1': 3,
                                     '舒适性,1': 11, '动力,-1': 15, '舒适性,0': 10, '空间,0': 25, '舒适性,-1': 9}
                # label = [y_word_index[x] for x in label.split('/')]
                # label = [y_word_index[label.split('/')[0]]]
                tensor_label = torch.Tensor(30).zero_().float()
                for i in label.split('/'):
                   tensor_label[y_wordsenti_index[i + "," + str(senti)]] = 1
                examples.append(data.Example.fromlist([text, tensor_label], fields))
        super(GrandDataset, self).__init__(examples, fields, **kwargs)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)


def torchLoad(config):

    TEXT = data.Field(sequential=True, fix_length=config.kwargs['padding_size'])
    LABEL = data.Field(sequential=True,use_vocab=False)

    train = GrandDataset(config.kwargs['raw_train_path'], text_field=TEXT, label_field=LABEL, config=config,test=False)
    val = GrandDataset(config.kwargs['raw_vali_path'], text_field=TEXT, label_field=LABEL, config=config,test=False)
    test = GrandDataset(config.kwargs['raw_test_path'], text_field=TEXT, label_field=None, config=config,test=True)

    cache = '../cache/'
    #读取W2V
    embedding_path = config.kwargs['embedding_path']
    print('load word2vec vectors from {}'.format(embedding_path))
    vectors = Vectors(name=embedding_path, cache=cache)
    vectors.unk_init = init.xavier_uniform_
    print('building {} vocabulary......'.format('Word'))
    TEXT.build_vocab(train, val, test, min_freq=1, vectors=vectors)


    train_iter = data.Iterator(dataset=train, batch_size=config.kwargs['batch_size'], sort=False,shuffle=True,repeat=False,
                                    device=-1)
    val_iter = data.Iterator(dataset=val, batch_size=config.kwargs['batch_size'], shuffle=False, sort=False, repeat=False,
                             device=-1)

    test_iter = data.Iterator(dataset=test, batch_size=config.kwargs['batch_size'], shuffle=False, sort=False, repeat=False,
                              device=-1)

    numerical_dict = TEXT.vocab.stoi

    return train_iter,val_iter,test_iter,TEXT.vocab.vectors,numerical_dict



# torchLoad(config)