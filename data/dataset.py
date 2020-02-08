import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from torch.utils import data
import pandas as pd
import numpy as np
from utils.lib import CharVectorizer
import random

class Ag_news(data.Dataset):
    def __init__(self, opt, train=True, test=False):
        if train:
            self.data = pd.read_csv(opt.train_data, header=None)
        if test:
            self.data = pd.read_csv(opt.test_data, header=None)
        self.data.columns = ['type', 'title', 'description']
        self.vertorizer = CharVectorizer(alphabet=opt.alphabet, maxlen=opt.maxlen)
        self.train = train
        self.test = test
        self.maxlen = opt.maxlen
        self.nclasses = len(pd.unique(self.data['type']))
        self.dim = len(opt.alphabet) + 1

    def __getitem__(self, index):#每次读取都要转化，花费太大|标题以及字符串连接,onehot
        label, title, txt = self.data.iloc[index]
        #lab = np.zeros(self.nclasses)
        #lab[int(label)-1] = 1#转onehot向量
        label = int(label) - 1
        txt = title + " " + txt
        if self.train:
            num = random.random()
            if num >= 0.5:
                txt = title + " " + title
        txt = self.vertorizer.transform(txt)
        vec_txt = np.zeros((self.maxlen, self.dim))
        for i in range(len(txt)):
            vec_txt[i][txt[i]] = 1  # 转onehot向量
        return label, vec_txt
    
    def __len__(self):
        return len(self.data)
