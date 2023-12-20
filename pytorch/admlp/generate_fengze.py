from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.nuscenes import NuScenes
import numpy as np
import pickle
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

def get_global_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def get_gt_trajectory(nusc, rec):
    n_output = 6
    gt_trajectory = []

    egopose_cur = get_global_pose(rec, nusc, inverse=True)

    for i in range(n_output + 1):
        next_token = rec['next']
        if next_token != '':
            next_sample = nusc.get('sample',rec['token'])

            egopose_future = get_global_pose(next_sample, nusc, inverse=False)

            egopose_future = egopose_cur.dot(egopose_future)
            theta = quaternion_yaw(Quaternion(matrix=egopose_future))

            origin = np.array(egopose_future[:3, 3])

            gt_trajectory.append([origin[0], origin[1], theta])
            rec = nusc.get('sample', next_token)
        else:
            next_sample = nusc.get('sample', rec['token'])
            egopose_future = get_global_pose(next_sample, nusc, inverse=False)
            egopose_future = egopose_cur.dot(egopose_future)
            theta = quaternion_yaw(Quaternion(matrix=egopose_future))

            origin = np.array(egopose_future[:3, 3])

            gt_trajectory.append([origin[0], origin[1], theta])
            break

    return gt_trajectory

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data2/NuScenes', verbose=False)
    with open('nuscenes_infos_val.pkl','rb') as f:
        vals = pickle.load(f)
    with open('nuscenes_infos_train.pkl','rb') as f:
        trains = pickle.load(f)

    for val in vals['infos']:
        token = val['token']
        sample = nusc.get('sample',token)
        val['gt_trajectory'] = get_gt_trajectory(nusc, sample)

    for train in trains['infos']:
        token = train['token']
        sample = nusc.get('sample',token)
        train['gt_trajectory'] = get_gt_trajectory(nusc, sample)

    with open('fengze_nuscenes_infos_train.pkl','wb') as f:
        pickle.dump(trains, f)
    with open('fengze_nuscenes_infos_val.pkl', 'wb') as f:
        pickle.dump(vals, f)
