import torch
from argparse import ArgumentParser

from stp3.planning_metrics import PlanningMetric
from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.utils.network import preprocess_batch
from stp3.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features
from stp3.trainer import TrainingModule

from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import pickle

def prepare_future_plan_labels(batch):
        labels = {}

        segmentation_labels = batch['segmentation'] 
        future_egomotion = batch['future_egomotion'] 
        gt_trajectory = batch['gt_trajectory'] 

        # gt trajectory
        labels['gt_trajectory'] = gt_trajectory
        
        # Warp labels to present's reference frame
        segmentation_labels_past = cumulative_warp_features(
            segmentation_labels[:, :3].float(), 
            future_egomotion[:, :3], 
            mode='nearest', spatial_extent=(50,50), 
        ).long().contiguous()[:, :-1] 
        segmentation_labels = cumulative_warp_features_reverse(
            segmentation_labels[:, (3 - 1):].float(), 
            future_egomotion[:, (3 - 1):],
            mode='nearest', spatial_extent=(50,50),
        ).long().contiguous() 
        labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)

        pedestrian_labels = batch['pedestrian']
        pedestrian_labels_past = cumulative_warp_features(
            pedestrian_labels[:, :3].float(),
            future_egomotion[:, :3],
            mode='nearest', spatial_extent=(50,50),
        ).long().contiguous()[:, :-1]
        pedestrian_labels = cumulative_warp_features_reverse(
            pedestrian_labels[:, (3 - 1):].float(),
            future_egomotion[:, (3 - 1):],
            mode='nearest', spatial_extent=(50,50),
        ).long().contiguous()
        labels['pedestrian'] = torch.cat([pedestrian_labels_past, pedestrian_labels], dim=1)
        return labels

   

def evaluate(final_traj_path=None, checkpoint_path=None, dataroot=None, online=False):
    future_second = 3
    n_present = 3
    #'/ssd2/djh/env/mask_for_occupancy/stp3_val/stp3_traj.pkl'
    pre_traj = open(final_traj_path,'rb')
    final_traj = pickle.load(pre_traj)
    device = torch.device('cuda:1' if online else 'cpu')
    metric_planning_val = []
    for i in range(future_second):
        metric_planning_val.append(PlanningMetric(2*(i+1)).to(device))

    if online == True:
        trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
        trainer.eval()
        trainer.to(device)
        model = trainer.model
        cfg = model.cfg
        cfg.GPUS = "[1]"
        cfg.BATCHSIZE = 1
        cfg.LIFT.GT_DEPTH = False
        cfg.DATASET.DATAROOT = dataroot
        cfg.DATASET.MAP_FOLDER = dataroot
        dataroot = cfg.DATASET.DATAROOT

        nworkers = cfg.N_WORKERS
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
        valdata = FuturePredictionDataset(nusc, 1, cfg)

        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False
        )
        for index, batch in enumerate(tqdm(valloader)):
            preprocess_batch(batch, device)
            labels = prepare_future_plan_labels(batch)
            occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                         labels['pedestrian'][:, n_present:].squeeze(2))
            for i in range(future_second):
                cur_time = (i+1)*2
                final = final_traj[batch["token"][0]]
                metric_planning_val[i](final[:,:cur_time].to(device), labels['gt_trajectory'][:,1:cur_time+1].to(device), occupancy[:,:cur_time].to(device))

    else:
        # 'stp3_val/segmentation_pedestrian_.pkl
        gt_traj = open('stp3_val/stp3_traj_gt.pkl','rb')
        gt_traj_traj = pickle.load(gt_traj)

        gt_occup = open('stp3_val/stp3_occupancy.pkl','rb')
        gt_traj_occup = pickle.load(gt_occup)

        token = open('stp3_val/filter_token.pkl','rb')
        token_filter = pickle.load(token)
        
        for token in tqdm(token_filter):
            final = final_traj[token]
            gt_trajectory =  torch.tensor(gt_traj_traj[token]['gt_trajectory']).unsqueeze(0)
            occupancy = gt_traj_occup[token]
            for i in range(future_second):
                cur_time = (i+1)*2
                metric_planning_val[i](final[:,:cur_time], gt_trajectory[:,1:cur_time+1], occupancy[:,:cur_time])

    
    results = {}
    for i in range(future_second):
        scores = metric_planning_val[i].compute()
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format(i+1)]=value.mean()
    for key, value in results.items():
        print(f'{key} : {value.item()}')

if __name__ == '__main__':

    online = False
    final_traj_path = 'stp3_val/stp3_traj_pre.pkl'
    if not online:
        evaluate(final_traj_path=final_traj_path, online = False)
    else:
        evaluate(final_traj_path=final_traj_path, checkpoint_path='ckpts/STP3_plan.ckpt', dataroot='data/nuscenes', online=True)
