import os
import random

from planner import VanillaPlanHead2
import torch
from torch import nn
from torch.nn import functional as F
import pickle
import numpy as np
import math
from torchmetrics import Metric
import copy
import math
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm,trange
from evaluate_for_mlp import run
def train_token():
    with open('fengze_nuscenes_infos_train.pkl','rb')as f:
        res=[]
        data=pickle.load(f)['infos']
        for ww in data:
            res.append(ww['token'])
        return res
def test_token():
    with open('stp3_val/filter_token.pkl','rb')as f:
        res=pickle.load(f)
        return res
class TokenDataset(Dataset):
    def __init__(self,train=True):
        super(TokenDataset, self).__init__()
        self.train=train
        self.tokens=train_token() if train else test_token()

    def __getitem__(self, item):
        return self.tokens[item]

    def __len__(self):
        return len(self.tokens)

def evaluate(model):
    dataset = TokenDataset(train=False)
    res = {}
    for i in trange(len(dataset)):
        token = [dataset[i]]
        pred = model.inference(token=token)
        res[token[0]] = pred
    with open('output_data.pkl','wb')as f:
        pickle.dump(res,file=f)
    return run()

def main(evaluation=False):
    model = VanillaPlanHead2(hidden_dim=512)
    optimizer = optim.AdamW(model.parameters(),lr=4e-6,weight_decay=1e-2)
    batch_size = 4
    dataset = TokenDataset()
    dataloader = DataLoader(dataset,batch_size,shuffle=True)
    if evaluation:
        model.load_state_dict(torch.load('rec/mlp.pth',map_location='cpu'))
        device = torch.device('cuda:0')
        model = model.to(device)
        evaluate(model)
        return
    device = torch.device('cuda:0')
    model = model.to(device)
    epochs = 6
    scheduler = MultiStepLR(optimizer,[2,4],gamma=0.2)
    evaluate(model)
    for epoch in trange(epochs):
        cnt=0
        model.train()
        for token in dataloader:
            cnt+=len(token)
            optimizer.zero_grad()
            loss = model(token=token)
            loss.backward()
            optimizer.step()
        scheduler.step()
        evaluate(model)



if __name__=='__main__':
    main(evaluation=True)
