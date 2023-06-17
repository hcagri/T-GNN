import os 
import os.path as osp
import copy

from t_gnn_lib import *

import torch 
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributions.multivariate_normal as torchdist

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False


dirname = os.path.dirname(__file__)
experiment_dir = os.path.join(dirname, 'experiments/test')
ex = Experiment("ceng502")
ex.observers.append(FileStorageObserver(experiment_dir))

ex.add_config('t_gnn_lib/test_config.yml')

@ex.automain
def main(_config, _run):

    run_path = os.path.join(experiment_dir, _run._id)
    sacred.commands.print_config(_run)
    os.makedirs(os.path.join(run_path, 'checkpoints'))
    model_args = _config['model']

    test_dset = TrajectoryDataset(
        osp.join(_config['data']['path'], 'test'),
        obs_len =_config['data']['seq_len_obs'],
        pred_len=_config['data']['seq_len_pred'],
        skip=1
        )
    
    test_loader = DataLoader(
        test_dset, 
        batch_size=1,
        shuffle=True,
        num_workers=2
    )

    model = T_GNN(
        n_stgcnn     = model_args['num_gcn'],
        n_txpcnn     = model_args['num_gcn'],
        input_feat   = model_args['input_size'],
        feat_dim     = model_args['feat_size'],
        output_feat  = model_args['output_size'],
        seq_len      = _config['data']['seq_len_obs'],
        pred_seq_len = _config['data']['seq_len_pred'],
        kernel_size  = model_args['kernel_size']
        ).cuda()

    model.load_state_dict(torch.load(_config['model_path']))

    ade, fde = evaluate(model, test_loader, _config)

    print(f"Results: \nade:{ade}, fde:{fde}")
