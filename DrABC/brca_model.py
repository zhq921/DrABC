from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dill
import blosc

def load_file(file_path):
    file_path = str(file_path)
    if file_path.endswith('.blosc'):
        with open(file_path, 'rb') as f:
            return dill.loads(blosc.decompress(f.read()))


class BRCA(nn.Module):
    def __init__(self, h1, h2, drop_prob):
        super(BRCA, self).__init__()
        self.fc1 = nn.Linear(25, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 2)
        self.dropout = nn.AlphaDropout(p=drop_prob)
        self.h1=h1
        self.h2=h2
        self.drop_prob=drop_prob
    @staticmethod
    def to_numpy(is_flatten, *args):
        if is_flatten:
            return [arg.detach().clone().numpy().flatten() for arg in args]
        else:
            return [arg.detach().clone().numpy() for arg in args]

    def forward(self, x):
        h1 = self.dropout(torch.selu(self.fc1(x)))
        h2 = self.dropout(torch.selu(self.fc2(h1)))
        h3 = self.fc3(h2)
        log_p = F.logsigmoid(h3)
        log_one_minus_p = F.logsigmoid(-h3)
        log_p0 = log_p[:,0]
        log_p1 = log_one_minus_p[:,0] + log_p[:,1]
        log_p2 = log_one_minus_p[:,0] + log_one_minus_p[:,1]
        return torch.stack([log_p0, log_p1, log_p2],dim=1), torch.stack([log_p[:,0], log_one_minus_p[:,0]],dim=1)

    def probability(self, x):
        self.eval()
        log_probs_multi, log_probs_binary = self.to_numpy(False, *self.forward(x))
        return np.exp(log_probs_multi), np.exp(log_probs_binary)

    def predict(self, x):
        probs_multi, probs_binary = self.probability(x)
        pred_multi, pred_binary=np.argmax(probs_multi,axis=1),np.argmax(probs_binary,axis=1)
        return pred_multi, pred_binary

    @staticmethod
    def to_torch_state_dict(numpy_state_dict,dtype, device):
        state_dict = OrderedDict([(key, torch.tensor(val,dtype=dtype,device=device)) for key, val in numpy_state_dict.items()])
        return state_dict


class BRCAForest(object):
    def __init__(self,state_forest, h1, h2, drop_prob, dtype, device):
        self.h1=h1
        self.h2=h2
        self.drop_prob=drop_prob
        self.brca_forest=[]
        self.dtype=dtype
        self.device=device
        for state_dict in state_forest:
            brca=BRCA(h1, h2, drop_prob)
            brca.load_state_dict(brca.to_torch_state_dict(state_dict,dtype,device))
            self.brca_forest.append(brca)

    def predict(self, x):
        probs_multi, probs_binary = self.probability(x)
        pred_multi=np.argmax(probs_multi,axis=1)
        pred_binary=np.argmax(probs_binary, axis=1)
        return pred_multi, pred_binary

    def probability(self, x):
        pred_multi_lst, pred_binary_lst = [], []
        tensor_x=torch.tensor(x, dtype=self.dtype, device=self.device)
        for brca in self.brca_forest:
            pred_multi, pred_binary = brca.predict(tensor_x)
            pred_multi_lst.append(pred_multi)
            pred_binary_lst.append(pred_binary)
        multi_count=pd.DataFrame(np.stack(pred_multi_lst,axis=1)).apply(lambda x:self.count_value(x,[0,1,2]),axis=1,result_type='expand')[[0,1,2]]
        binary_count=pd.DataFrame(np.stack(pred_binary_lst,axis=1)).apply(lambda x:self.count_value(x,[0,1]),axis=1,result_type='expand')[[0,1]]
        probs_multi=multi_count.values/multi_count.sum(axis=1).values.reshape(-1,1)
        probs_binary=binary_count.values/binary_count.sum(axis=1).values.reshape(-1,1)
        return probs_multi, probs_binary

    @staticmethod
    def count_value(x,labels):
        cv=dict(zip(*np.unique(x,return_counts=True)))
        for val in labels:
            if val not in cv:
                cv[val]=0
        return cv
