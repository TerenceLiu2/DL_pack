# coding=utf-8
'''

@author:Teren
构建W2V语料以及构建W2V

'''


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
from Config import Config

# overwrite = True
config = Config(
                corpus_path='../data/all.csv',
                raw_train_path = '../data/train.csv',
                raw_test_path = '../data/test_public.csv',
                )


def CreateCharCorpus(config):

    count_dict = {}
    cot = 0
    corpus = []
    df_train = pd.read_csv(config.kwargs['raw_train_path'],usecols=['content']).drop_duplicates(subset=['content'])
    df_test = pd.read_csv(config.kwargs['raw_test_path'],usecols=['content'])

    df_all = pd.concat([df_train,df_test])
    for line in df_all.values:
        tmp_line = []
        for c in line[0]:
            cot += 1
            tmp_line.append(c)
            if c not in count_dict:
                count_dict[c] = 0
            else:
                count_dict[c] += 1
        corpus.append(' '.join(tmp_line))

    print("Total sum: %d"%cot)

    with open(config.kwargs['corpus_path'],'w') as wf:
        for l in corpus:
            wf.write(l+"\n")



def PreProcess():

    df_train = pd.read_csv('../data/test_word_char.csv',usecols=['content'])
    df_test = pd.read_csv('../data/train_word_char.csv',usecols=['content'])

    df_all = pd.concat([df_train,df_test])

    df_all.to_csv("../data/all.csv",index=False)



    pass

def W2V_model(config):
    model = Word2Vec(
        LineSentence(config.kwargs['corpus_path']),
        size=config.kwargs['w2v_dim'],
        window=config.kwargs['w2v_window'],
        min_count=1,
        workers=2
    )
    model.wv.save_word2vec_format("../data/%d_%d_wc.model"%(config.kwargs['w2v_dim'],config.kwargs['w2v_window']))

# CreateCharCorpus(config)
# PreProcess()
W2V_model(config)
