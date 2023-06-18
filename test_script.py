import os 
import os.path as osp
import json

from t_gnn_lib import *

import torch 
from torch.utils.data import DataLoader

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False


dirname = os.path.dirname(__file__)
experiment_dir = os.path.join(dirname, 'experiments/test')
ex = Experiment("ceng502")
ex.observers.append(FileStorageObserver(experiment_dir))


@ex.config
def my_config():
    """ Please provide a run ID and select which model parameters to use """
    
    exp_run_id = 3
    which_epoch = 250


    exp_path = osp.join('experiments/training', str(exp_run_id))
    with open(osp.join(exp_path, 'config.json'), 'r') as f_:
        config = json.load(f_)
    config['model_path'] = osp.join(exp_path, 'checkpoints', f'epoch_{which_epoch}.pth')


@ex.automain
def main(_config, _run):

    config = _config['config']

    run_path = os.path.join(experiment_dir, _run._id)
    sacred.commands.print_config(_run)
    os.makedirs(os.path.join(run_path, 'checkpoints'))
    model_args = config['model']

    test_dset = TrajectoryDataset(
        osp.join(config['data']['path'], 'test'),
        obs_len =config['data']['seq_len_obs'],
        pred_len=config['data']['seq_len_pred'],
        skip=1
        )
    
    test_loader = DataLoader(
        test_dset, 
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    model = T_GNN(
        n_stgcnn     = model_args['num_gcn'],
        n_txpcnn     = model_args['num_gcn'],
        input_feat   = model_args['input_size'],
        feat_dim     = model_args['feat_size'],
        output_feat  = model_args['output_size'],
        seq_len      = config['data']['seq_len_obs'],
        pred_seq_len = config['data']['seq_len_pred'],
        kernel_size  = model_args['kernel_size']
        ).cuda()

    model.load_state_dict(torch.load(config['model_path']))

    ade_r, fde_r = evaluate(model, test_loader, config)

    print(f"Results: \nade:{ade_r}, fde:{fde_r}")
