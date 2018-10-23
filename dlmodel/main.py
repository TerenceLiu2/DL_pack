# -*- coding: utf-8 -*-

'''

@author: Teren
深度学习主函数，设置参数选择模型等
待改进点：
    1. 当F1有提升的时候才保存Epoch
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
from Config import Config
from preprocessing import data_preprocessing
import pandas as pd
from dlmodel.TextCNN import TextCNN
from dlmodel.LSTM import LSTM
from dlmodel.AttLSTM import AttLSTM
from dlmodel.RCNN import RCNN
from dlmodel.Inception import InCNN


def predict_proba(config,test_iter):
    '''
    :param config: 配置文件 - submit_path - raw_test_path - model_path - threshold
    :param test_iter: Test Iterator
    :return:
    '''

    count = 0
    submit_file = open(config.kwargs['submit_path'], "w")
    test_id = pd.read_csv(config.kwargs['raw_test_path'])['content_id'].values

    for batch in test_iter:
        test_data = batch.text
        model = torch.load(config.kwargs['model_path'])
        out = model(test_data)
        test_data = test_data.numpy().T
        pred_y = torch.Tensor(len(out), 30).zero_().long()

        for idx, o in enumerate(out):
            # 标签文本存在就做标注
            # for k in labelnum_index_dict.keys():
            #     if k in test_data[idx].tolist():
            #         index = (labelnum_index_dict[k] - 1) * 3
            #         tmp_max = max(o[index:index + 3].tolist())
            #         pred_y[idx][index + o[index:index + 3].tolist().index(tmp_max)] = 1
            if max(o) < config.kwargs['threshold']:
                if (max(pred_y[idx]).data == 0):
                    pred_y[idx][o.tolist().index(max(o))] = 1
                else:
                    pass
            for idy in range(0, 30, 3):
                tmp_max = max(o[idy:idy + 3].tolist())
                if tmp_max >= config.kwargs['threshold']:
                    pred_y[idx][idy + o[idy:idy + 3].tolist().index(tmp_max)] = 1

        # print(pred_y.shape)

        for index,item in enumerate(pred_y.data.numpy()):
            print(test_id[count])
            for idy,label in enumerate(item):
                if label != 0:
                    print(y_index_word[idy])
                    submit_file.write(test_id[count]+","+y_index_word[idy]+","+"\n")
            count += 1

        # print(pred_y.data.numpy())

config = Config(
                batch_size=128,
                epoch=80,
                label_num=30,
                padding_size=400,
                sentence_max_size=100, #自定义句子最大长度，CNN Pooling参数
                word_num=26755, # Word2Vec的值 Word+Char
                # word_num=19789, # Word
                aug=False, #是否做数据打乱以及Drop
                hidden_size=100, #隐藏层Layers 100 for RNN 400 for CNN
                num_layers=2, #LSTM层数 1 for Attention 2 for LSTM
                learning_rate=1e-3,
                threshold=0.1,  # 阈值
                model_path='../model/0.65_30_Inception_epoch16.ckpt', # Test用Model
                submit_path='../result/0.65_30_Inception_epoch16_0.577.csv',#生成文件路径
                raw_train_path = '../data/train_train_word_char.csv',
                raw_test_path = '../data/test_word_char.csv',
                raw_vali_path = '../data/vali_train_word_char.csv',
                embedding_path = '../data/300_5_wc.model', #W2V的位置
               )

#建立输出字典
y_word_index = {'动力': 1, '价格': 2, '安全性': 7, '操控': 4, '空间': 10, '内饰': 8, '外观': 9, '舒适性': 5, '油耗': 3,
                '配置': 6}
y_wordsenti_index = {'动力,1': 17, '安全性,-1': 21, '外观,0': 7, '油耗,0': 1, '价格,0': 19, '操控,0': 28, '配置,-1': 12, '安全性,0': 22, '安全性,1': 23, '配置,1': 14, '内饰,1': 5, '空间,-1': 24, '价格,1': 20, '价格,-1': 18, '外观,-1': 6, '空间,1': 26, '外观,1': 8, '配置,0': 13, '内饰,0': 4, '动力,0': 16, '操控,-1': 27, '操控,1': 29, '油耗,-1': 0, '油耗,1': 2, '内饰,-1': 3, '舒适性,1': 11, '动力,-1': 15, '舒适性,0': 10, '空间,0': 25, '舒适性,-1': 9}
y_index_word = {v: k for k, v in y_wordsenti_index.items()}

"""
Load Data
"""

train_iter,vali_iter,test_iter,vectors,numerical_dict = data_preprocessing.torchLoad(config)

# labelnum_index_dict = {numerical_dict[k]:y_word_index[k] for k in y_word_index.keys()}


"""
Init
"""

# model = RCNN(config,vectors)
# model = TextCNN(config,vector=vectors)
model = LSTM(config,vectors)
# model = InCNN(config,vectors)
#  权重越小对应类的损失对总体损失的影响越小
criterion = nn.BCELoss()
optimizer = model.get_optimizer()
# predict_proba(config,test_iter)
# exit(0)

"""
Model Start
"""

count = 0
loss_sum = 0

for epoch in range(config.kwargs['epoch']):

    print("=================epoch %d ==================" % epoch)

    for batch in train_iter:
        # print(data,label)
        data = batch.text
        label = batch.label

        # print(data)

        data = autograd.Variable(data.long())

        # Question
        out = model(data)


        label = torch.from_numpy(label.numpy().T)
        label = autograd.Variable(label)

        loss = criterion(out.float(), label.float())


        loss_sum += loss.data[0]
        count += 1

        # print(loss_sum,count)
        """
        计算F1
        """

        if count % 10 == 0:
            # Question

            Tp = 0
            Fp = 0
            Fn = 0

            count_y = {}
            count_y_acc = {}
            count_y_rec = {}

            for vali_batch in vali_iter:

                vali_data = vali_batch.text
                vali_label = vali_batch.label.numpy().T

                out = model(autograd.Variable(vali_data.long()))

                vali_data = vali_data.numpy().T

                pred_y = torch.Tensor(len(out), 30).zero_().long()

                print(out[18:20])

                for idx, o in enumerate(out):

                    # 标签文本存在就做标注
                    # for k in labelnum_index_dict.keys():
                    #     if k in vali_data[idx].tolist():
                    #         index = (labelnum_index_dict[k]-1)*3
                    #         tmp_max = max(o[index:index + 3].tolist())
                    #         pred_y[idx][index + o[index:index + 3].tolist().index(tmp_max)] = 1
                    if max(o) < config.kwargs['threshold']:
                        if (max(pred_y[idx]).data==0):
                            pred_y[idx][o.tolist().index(max(o))] = 1
                        else:
                            pass
                    for idy in range(0,30,3):
                        tmp_max = max(o[idy:idy+3].tolist())
                        if  tmp_max >= config.kwargs['threshold']:
                            pred_y[idx][idy + o[idy:idy+3].tolist().index(tmp_max)] = 1

                # print(pred_y)
                # print(y_vali)

                for idx, line in enumerate(vali_label):
                    for idy, l in enumerate(line.tolist()):
                        if idy not in count_y:
                            count_y[idy] = 1
                            count_y_acc[idy] = 0
                            count_y_rec[idy] = 0
                        if l == 0:
                            if pred_y[idx][idy] != l:
                                Fp += 1
                                count_y[idy] += 1
                        if l == 1:
                            if pred_y[idx][idy] != l:
                                Fn += 1
                                count_y_rec[idy] += 1
                            else:
                                Tp += 1
                                count_y_acc[idy] += 1

            Pre = float(Tp)/(Tp+Fp)
            Rec = float(Tp)/(Tp+Fn)


            F1 = (float(2)*Pre*Rec)/(Pre+Rec)

            # accuracy = float((pred_y == label.data.numpy()).astype(int).sum()) / float(label.size(0))
            print("============total loss F1 ============")
            # print("The loss is: %.5f accuracy is %.5f" % (loss_sum / 10, accuracy))
            print("The loss is: %.5f F1 is %.5f Pre is %.5f Rec is %.5f" % (loss_sum / 10, F1,Pre,Rec))

            print("============label acc ============")


            for k in count_y.keys():
                print(y_index_word[k],"acc: %.4f rec: %.4f"%(float(count_y_acc[k])/(count_y[k]+count_y_acc[k]),float(count_y_acc[k])/(count_y_rec[k]+count_y_acc[k])))


            loss_sum = 0
            count = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # save the model in every epoch
    torch.save(model,'../cache/epoch{}.ckpt'.format(str(epoch)))