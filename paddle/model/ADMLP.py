import paddle 
from paddle import nn
from paddle.nn import functional as F

import math
import copy
import pickle
import numpy as np

from skimage.draw import polygon
from typing import Any, Dict, List, Optional, Tuple, Union
from paddle.nn.initializer import Assign

 
def count_layers(obj):
    if isinstance(obj, list):
        return 1 + max(count_layers(x) for x in obj)
    else:
        return 0

class ADMLP(nn.Layer):
    def __init__(
        self,
        num_heads=8,
        hidden_dim=256,
        dropout=0.1,
        activation='relu',
        ffn_channels=256,
        future_frames=6
    ):
        super().__init__()
        self.velocity_dim=3
        self.past_frame=5
        self.plan_head=nn.Sequential(
							nn.Linear(self.velocity_dim*(self.past_frame+2), 512),
							nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512,7*3)
						)
    def forward(self, input, token):
        device=input.place
        dtype=input.dtype
        if isinstance(input,list):
            input=input[0]
        input=0
        
        # if 'token' in kwargs:
        if not hasattr(self,'ad'):
            self.ad=pickle.load(open('data_nuscene.pkl','rb'))

        tokens=token
        velocitys=[]
        for j,token in enumerate(tokens):
            assert token in self.ad
            cur_info=[]
            key=list(self.ad[token])
            key.sort()
            for k in key:
                ele=self.ad[token][k]
                if count_layers(ele)==2:
                    cur_info+=ele
                else:
                    cur_info.append(ele)
            cur_info=paddle.to_tensor(cur_info, place=device).astype(dtype).flatten().unsqueeze(-1)
            velocitys.append(cur_info)
        velocitys=paddle.concat(velocitys,axis=-1).transpose([1,0]) #bs,21
        input=velocitys
        
        input=self.plan_head(input) # bs,21
        waypoints = {}
        for i in range(1,8):
            waypoints['x{}'.format(i)]=input[:,3*(i-1):3*i].unsqueeze(-1)
        return waypoints
    

if __name__ == '__main__':
    ############# pytorch pth -> paddle pdparams ###################
    # import torch
    # params = torch.load('mlp.pth')
    # paddleparams = dict()
    # for key, value in params.items():
    #     value = value.cpu().numpy()
    #     value = paddle.to_tensor(value, dtype='float32')
    #     paddleparams[key] = value
    # paddle.save(paddleparams, 'mlp.pdparams')

    ############## load pdparams ##################
    params_paddle = paddle.load('mlp.pdparams')

    ############## model ##################
    input = paddle.ones([1,512,180,180])
    deviceinput = input.place
    input = input.cuda()
    model = ADMLP(hidden_dim=512)

    ############## update params ##################
    state_dict_model = dict()
    for key, value in params_paddle.items():
        if key[11:] == 'plan_head.0.weight' or key[11:] == 'plan_head.2.weight' or key[11:] == 'plan_head.4.weight':
            state_dict_model[key[11:]] = value.T
        else: 
            state_dict_model[key[11:]] = value
    model.set_state_dict(state_dict_model)

    ############## load pkl ##################
    pickle_file = open('filter_token.pkl','rb')
    tokenpkl = pickle.load(pickle_file)

    ############## infer ##################
    val_data_paddle_pkl = dict()
    i = 0
    for token in tokenpkl:
        print(i)
        token=[token]
        predict_dict = model(input,token=token)
        pred = paddle.concat((predict_dict['x2'],predict_dict['x3'],predict_dict['x4'],predict_dict['x5'],predict_dict['x6'],predict_dict['x7']), axis=-1)
        pred = pred.transpose([2,1,0]).squeeze(-1).unsqueeze(0)
        pred = paddle.to_tensor(pred, place=deviceinput).numpy()
        token = token[0]
        val_data_paddle_pkl[token] = pred
        i+=1

    f = open("val_data.pkl","wb")
    pickle.dump(val_data_paddle_pkl, f)
