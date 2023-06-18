import os 
import os.path as osp

from t_gnn_lib import *

import torch 
import torch.optim as optim
from torch.utils.data import DataLoader

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False


dirname = os.path.dirname(__file__)
experiment_dir = os.path.join(dirname, 'experiments/training_v2')
ex = Experiment("ceng502")
ex.observers.append(FileStorageObserver(experiment_dir))

ex.add_config('t_gnn_lib/train_config.yml')

@ex.automain
def main(_config, _run):

    run_path = os.path.join(experiment_dir, _run._id)
    sacred.commands.print_config(_run)
    os.makedirs(os.path.join(run_path, 'checkpoints'))
    model_args = _config['model']

    source_dset = TrajectoryDataset_Social(
        osp.join(_config['data']['path'], 'train'),
        obs_len =_config['data']['seq_len_obs'],
        pred_len=_config['data']['seq_len_pred'],
        skip=1
        )
    
    source_loader = DataLoader(
        source_dset, 
        batch_size=1,
        shuffle=True,
        num_workers=2
    )

    target_dset = TrajectoryDataset_Social(
        osp.join(_config['data']['path'], 'val'),
        obs_len =_config['data']['seq_len_obs'],
        pred_len=_config['data']['seq_len_pred'],
        skip=1
        )
    
    target_loader = DataLoader(
        target_dset, 
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

    print(model)

    optimizer = optim.Adam(model.parameters(), lr= _config['training']['lr'])
    loss_fn = bivariate_loss

    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e16}

    for epoch in range(_config['training']['num_epochs']):
        train_loss = train_v2(model, optimizer, loss_fn, source_loader, target_loader, epoch, _config)
        # val_loss = validate(model, loss_fn, target_loader, epoch, _config)

        metrics['train_loss'].append(train_loss)
        # metrics['val_loss'].append(val_loss)

        if epoch == _config['training']['change_lr']:
            optimizer.param_groups[0]['lr'] = 0.0005
        
        # if val_loss < constant_metrics['min_val_loss']:
        #     constant_metrics['min_val_epoch'] = epoch 
        #     constant_metrics['min_val_loss'] = val_loss

        #     torch.save(model.state_dict(), os.path.join(run_path, 'checkpoints', f'epoch_{epoch+1}.pth'))
        
        # else:
        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(), os.path.join(run_path, 'checkpoints', f'epoch_{epoch+1}.pth'))


    print('\n\n END of TRAINING \n\n')
    print(constant_metrics)
