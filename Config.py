#coding=utf8

"""
@author: Teren
项目全局配置
待解决：
1. 可自定义更多全局参数
"""



from termcolor import *
import os

class Config:

    def __init__(self,**kwargs):

        self.kwargs = kwargs
        self.root_dir = '/media/teren/My Passport/Code/BDCI/'
        self.cache_dir = '/media/teren/My Passport/Code/BDCI/cache'
        print(self.cache_dir)

        if 'w2v_dim' not in kwargs:
            print(colored('Warning!!! Default w2v_dim 300', 'yellow'))
            self.kwargs['w2v_dim'] = 300
        else:
            self.kwargs['w2v_dim'] = kwargs['w2v_dim']

        if 'w2v_window' not in kwargs:
            print(colored('Warning!!! Default w2v_window 5', 'yellow'))
            self.kwargs['w2v_window'] = 5
        else:
            self.kwargs['w2v_window'] = kwargs['w2v_window']





