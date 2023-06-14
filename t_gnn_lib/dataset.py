import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time


def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    return NORM
                
def seq_to_graph(seq_,seq_rel):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    # For each time step, we must have a Graph. So total we will have seq_len graphs: 8 default.
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes)) # 8 graphs with NxN. N is the number of the pedestrians in a given scene
    for s in range(seq_len):
        # So create seg_len number of graphs
        # step_: Stores the positions of the pedestrians for each time step. shape: (num_peds, 2 (x,y))
        step_ = seq_[:,:,s] # The shape of seq_: (num_peds, 2(x,y), seq_len). Take the each time stemp in a sequence iteratively.
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            # Iterate for each pedestrian 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 0 
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm 
                A[s,k,h] = l2_norm # Undirected graphs has symetric adjacency matrix
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)



def decentralization(seg_list_rel_temp, obs_len):
    # seg_list_rel_temp -> shape: (num_seq, 2, seg_len)
    seg_list_rel_temp = np.concatenate(seg_list_rel_temp, axis=0)
    pos_at_t_obs = seg_list_rel_temp[:, :, obs_len] # Get the x and y locations of the last observation time, T_obs

    # Calculate the mean x and y locations
    mean_x = np.mean(pos_at_t_obs[:, 0])
    mean_y = np.mean(pos_at_t_obs[:, 1])
    mean_position = np.array([mean_x, mean_y])
    mean_position = np.reshape(mean_position, (1, 2, 1))

    seg_list_rel = seg_list_rel_temp - mean_position

    return seg_list_rel



def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    ''' Source: https://github.com/agrimgupta92/sgan/blob/master/sgan/data/trajectories.py '''
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = False):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            seg_list_rel_temp = [] # sequence list only for given scene
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # How many pederstrians are exists in current sequence
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))

                # curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len)) # shape: (num_peds, 2 (x,y), sequence_len)
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))     # shape: (num_peds, 2 (x,y), sequence_len)
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # Iterate over each pedestrian
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        # For current pedestrian, if the length of it is not equal to squence length then discard it. However it may include missing detections in between. 
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) # Only take the x and y locations of pedestrian with id=ped_id 
                    curr_ped_seq = curr_ped_seq # shape: (2, sequence_len) [x;y]
                    # Make coordinates relative
                    # rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    # Instead of using x,y locations. Store the difference between x and y locations. And make them relative.
    
                    # rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1] 
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    # curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered]) # Take only valid pedestrians who can be tracked as the duration of sequence length.
                    seg_list_rel_temp.append(curr_seq[:num_peds_considered])
            ''' Compute Relative Locations using decentralization operation '''
            seq_rel = decentralization(seg_list_rel_temp, self.obs_len)
            seq_list_rel.append(seq_rel)


        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0) # shape: (num_seq, 2, seg_len)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.x = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() 
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ] # These start end sequences represents the number of pedestrians in each time subset. 0->3 means there are 4 pedestrians occur at the same time for the frames in between 0-20 for example.
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :]) # v_: shape (seq_len, num_peds, (x,y)) | a_: shape (seq_len, num_peds, num_peds)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :])
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out
