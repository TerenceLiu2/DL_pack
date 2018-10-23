#coding=utf8
"""

@author：Teren
对数据进行分词以及划分训练验证集

"""

import pandas as pd
import jieba


def splitVali(filepath):
    """
    切分训练验证集
    :return:
    """
    filename = filepath.split('/')[2]
    df_train = pd.read_csv(filepath)

    id_content_dict = {}
    id_label_dict = {}
    id_senti_dict={}

    for line in df_train.values:

        if line[0] not in id_label_dict:
            id_label_dict[line[0]] = []

        id_content_dict[line[0]] = line[1]
        id_label_dict[line[0]].extend(line[2].split())
        id_senti_dict[line[0]] = str(line[3])

    cutpoiont = int(len(df_train) * 0.8)
    cot = 0

    wf1 = open('../data/%s'%("train_"+filename), 'w')
    wf2 = open('../data/%s'%("vali_"+filename), 'w')
    wf1.write("content_id,content,subject,sentiment_value"+"\n")
    wf2.write("content_id,content,subject,sentiment_value"+"\n")


    for k in id_label_dict.keys():
        if cot < cutpoiont:
            cot += 1
            wf1.write(
                    k + "," + id_content_dict[k] + "," + '/'.join(id_label_dict[k]) + "," + id_senti_dict[k] + "\n")
        else:
            wf2.write(
                    k + "," + id_content_dict[k] + "," + '/'.join(id_label_dict[k]) + "," + id_senti_dict[k] + "\n")

    wf1.close()
    wf2.close()

def segData():
    """
    中文分词
    :return:
    """
    df = pd.read_csv('../data/test_public.csv')

    for line in df.itertuples():
        df.loc[line[0], 'content'] = ' '.join(jieba.lcut(df.loc[line[0], 'content'].strip()))

    df.to_csv('../data/test_public_seg.csv',index=False)

def segCharData():
    """
    切分为Char
    :return:
    """
    df = pd.read_csv('../data/train.csv')

    for line in df.itertuples():
        df.loc[line[0], 'content'] = ' '.join([x for x in (df.loc[line[0], 'content'].strip())])

    df.to_csv('../data/train_char.csv',index=False)

def buildCharWordData(word_path,char_path,out_path):
    """
    构造词字联合输入向量
    :return:
    """

    df_w = pd.read_csv(word_path)
    df_c = pd.read_csv(char_path)

    for line in df_c.itertuples():
        df_w.loc[line[0],'content'] = df_w.loc[line[0],'content'] + df_c.loc[line[0],'content']
        # print(df_w.loc[line[0],'content'])

    df_w.to_csv(out_path,index=False)




# segData()
# segCharData()
splitVali("../data/train_char.csv")
# buildCharWordData("../data/test_public_seg.csv","../data/test_public_char.csv",'../data/test_word_char.csv')
# buildCharWordData("../data/train_seg.csv","../data/train_char.csv",'../data/train_word_char.csv')
