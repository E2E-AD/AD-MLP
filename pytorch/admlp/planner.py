from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import pickle
import numpy as np
import math
from torchmetrics import Metric
import copy
import math
from skimage.draw import polygon

__all__ = ["VanillaPlanHead2"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


def in_same_segment(x, y):
    x_floor = math.floor(x / 0.5)
    y_floor = math.floor(y / 0.5)
    return x_floor == y_floor


def l2_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    valid = targets > (-5e3)
    return F.mse_loss(inputs[valid], targets[valid], reduction=reduction)


def l1_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float().flatten()
    targets = targets.float().flatten()
    valid = targets > (-5e3)
    ww = 0.5
    for i in range(len(inputs)):
        if valid[i]:
            if in_same_segment(inputs[i], targets[i]):
                inputs[i] = inputs[i] * ww
                targets[i] = targets[i] * ww
    return F.l1_loss(inputs[valid], targets[valid], reduction=reduction)




class PlanningMetric_3:
    def __init__(
            self,
            n_future=6,
    ):
        X_BOUND = [-54.0, 54.0, 0.6]  #  Forward
        Y_BOUND = [-54.0, 54.0, 0.6]  # Sides
        Z_BOUND = [-10.0, 10.0, 20.0]
        dx, bx, _ = gen_dx_bx(X_BOUND, Y_BOUND, Z_BOUND)
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)

        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(X_BOUND, Y_BOUND, Z_BOUND)
        self.bev_dimension = self.bev_dimension.numpy()
        WIDTH = 1.85
        HEIGHT = 4.084
        self.W = WIDTH
        self.H = HEIGHT

        self.n_future = n_future

    def evaluate_single_coll(self, traj, segmentation):
        '''
        gt_segmentation
        traj: torch.Tensor (n_future, 2)
        segmentation: torch.Tensor (n_future, 200, 200) -> (n_future, x, 2)
        '''
        n_future, _ = traj.shape
        n_future = min(len(segmentation), n_future)
        traj = traj.clone()
        traj = traj * torch.tensor([-1, 1], device=traj.device)
        occidxs = copy.deepcopy(segmentation)
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]

        rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        trajs = traj.view(traj.shape[0], 1, 2)
        trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]  # can also change original tensor
        trajs = trajs / self.dx.to(trajs.device)
        rc = torch.tensor(rc).to(trajs.device).to(trajs.dtype)
        trajs = trajs + rc  # (n_future, 32, 2)

        # trajs is converted to pixel index
        r = trajs[:, :, 0].to(torch.int32)
        r = torch.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs[:, :, 1].to(torch.int32)
        c = torch.clip(c, 0, self.bev_dimension[1] - 1)

        distance = []
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = (rr >= 0) & (cc >= 0) & (rr < self.bev_dimension[0]) & (cc < self.bev_dimension[1])
            rr = rr[I].to(torch.float32)
            cc = cc[I].to(torch.float32)
            rr = rr.mean(dim=0)
            cc = cc.mean(dim=0)
            cur_occ = torch.tensor(occidxs[t]).to(rr.dtype).to(rr.device)
            # min distance to collide
            cur_occ[:, 0] = cur_occ[:, 0] - rr
            cur_occ[:, 1] = cur_occ[:, 1] - cc
            cur_occ = (cur_occ * cur_occ).sum(dim=-1)
            values, indices = torch.topk(cur_occ, min(3, len(cur_occ)), largest=False)
            distance.append(values.mean())
        return sum(distance) / n_future


def count_layers(obj):
    if isinstance(obj, list):
        return 1 + max(count_layers(x) for x in obj)
    else:
        return 0


class VanillaPlanHead2(nn.Module):
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
        self.velocity_dim = 3
        self.past_frame = 5
        self.plan_head = nn.Sequential(
            nn.Linear(self.velocity_dim * (self.past_frame + 2), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            # nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(512, 7 * 3)
        )

    def warp_gt(self, gts, device):
        res=[]
        for gt in gts:
            gt=gt.tolist()
            while len(gt)<7:
                gt.append([-1e4]*3)
            res.append(gt)
        return torch.tensor(res,device=device)


    def forward(self, **kwargs):
        device = torch.device('cuda:0')
        dtype = torch.float32
        gts=[]
        if 'token' in kwargs:
            if not hasattr(self, 'ad'):
                self.ad = pickle.load(open('stp3_val/data_nuscene.pkl', 'rb'))
            tokens = kwargs['token']
            velocitys = []
            for j, token in enumerate(tokens):
                assert token in self.ad
                cur_info = []
                key = list(self.ad[token])
                key.sort()
                for k in key:
                    if k=='gt':continue
                    ele = self.ad[token][k]
                    if count_layers(ele) == 2:
                        cur_info += ele
                    else:
                        cur_info.append(ele)
                cur_info = torch.tensor(cur_info).to(device).to(dtype).flatten().unsqueeze(-1)
                velocitys.append(cur_info)
                gts.append(self.ad[token]['gt'])

        velocitys = torch.cat(velocitys, dim=-1).permute(1, 0)  # bs,21
        input = velocitys

        input = self.plan_head(input)  # bs,21
        waypoints = {}
        for i in range(1, 8):
            waypoints['x{}'.format(i)] = input[:, 3 * (i - 1):3 * i].unsqueeze(-1)
        gts=self.warp_gt(gts,device)
        return self.loss(waypoints,gts)

    def loss(self, predict_dict, gt_trajectory):
        gt_trajectory = gt_trajectory.permute(1, 2, 0)  # 7,3,bs

        predict = torch.cat((predict_dict['x1'], predict_dict['x2'], predict_dict['x3'], predict_dict['x4'],
                             predict_dict['x5'], predict_dict['x6'], predict_dict['x7']), dim=-1)
        predict = predict.permute(2, 1, 0)
        loss = l1_loss(predict, gt_trajectory)
        loss_dict = {'default_loss': loss}

        return sum(loss_dict.values())

    def inference(self,**kwargs):
        device = torch.device('cuda:0')
        dtype = torch.float32
        if 'token' in kwargs:
            if not hasattr(self, 'ad'):
                self.ad = pickle.load(open('stp3_val/data_nuscene.pkl', 'rb'))
            tokens = kwargs['token']
            velocitys = []
            for j, token in enumerate(tokens):
                assert token in self.ad
                cur_info = []
                key = list(self.ad[token])
                key.sort()
                for k in key:
                    if k == 'gt': continue
                    ele = self.ad[token][k]
                    if count_layers(ele) == 2:
                        cur_info += ele
                    else:
                        cur_info.append(ele)
                cur_info = torch.tensor(cur_info).to(device).to(dtype).flatten().unsqueeze(-1)
                velocitys.append(cur_info)

        velocitys = torch.cat(velocitys, dim=-1).permute(1, 0)  # bs,21
        input = velocitys

        input = self.plan_head(input).detach().cpu().numpy()  # bs,21
        res = []
        for i in range(2, 8):
            res.append(input[:, 3 * (i - 1):3 * i])
        res = np.concatenate(res,axis=0)
        res = res.reshape((1,*res.shape))
        return res




if __name__ == '__main__':

    pass


