import torch 
import numpy as np
from .utils import *

def train(model, 
          optimizer, 
          loss_fn, 
          source_loader, 
          target_loader,
          epoch,
          _config
         ):
    
    batch_size = _config['training']['batch_size']

    model.train()
    loss_batch = 0 
    num_iter = 0
    loss = 0 
    loader_len = min(len(source_loader), len(target_loader))
    turn_point =int(loader_len/batch_size)*batch_size+ loader_len%batch_size -1
    first_loss=True

    for i_, (batch_source, batch_target) in enumerate(zip(source_loader, target_loader)):
        num_iter += 1

        batch_source = [tensor.cuda() for tensor in batch_source]
        batch_target = [tensor.cuda() for tensor in batch_target]

        V_obs_source, A_obs_source, V_pred_gt, A_pred_gt = batch_source
        V_obs_target, A_obs_target, _, _ = batch_target

        A_obs_source += torch.eye(A_obs_source.shape[3]).cuda()
        A_obs_source = A_obs_source/torch.sum(A_obs_source, dim=3, keepdim=True)

        A_obs_target += torch.eye(A_obs_target.shape[3]).cuda()
        A_obs_target = A_obs_target/torch.sum(A_obs_target, dim=3, keepdim=True)

        optimizer.zero_grad()

        V_pred, _, L_align = model(V_obs_source, A_obs_source.squeeze(), V_obs_target, A_obs_target.squeeze())


        V_pred_gt = V_pred_gt.squeeze()
        A_pred_gt = A_pred_gt.squeeze()
        V_pred = V_pred.squeeze()


        if num_iter % batch_size!=0 and i_ != turn_point:
            if first_loss:
                loss = loss_fn(V_pred, V_pred_gt) + L_align * _config['training']['lambda'] 
                first_loss = False
            else:
                loss += loss_fn(V_pred, V_pred_gt) + L_align * _config['training']['lambda'] 
        
        else:
            loss = loss / batch_size
            first_loss=True

            loss.backward()
            optimizer.step()
            loss_batch += loss.item()
            print('TRAIN:','\t Epoch:', epoch,'\t Loss:',loss_batch/num_iter)


    return loss_batch/num_iter



def validate(model, 
             loss_fn,  
             target_loader,
             epoch,
             _config
            ):
    batch_size = _config['training']['batch_size']

    model.eval()
    loss_batch = 0 
    num_iter = 0
    loss = 0 
    loader_len = len(target_loader)
    turn_point =int(loader_len/batch_size)*batch_size+ loader_len%batch_size -1
    first_loss=True

    with torch.no_grad():
        for i_, batch_target in enumerate(target_loader):
            num_iter += 1

            batch_target = [tensor.cuda() for tensor in batch_target]

            V_obs_target, A_obs_target, V_pred_gt, A_pred_gt = batch_target

            A_obs_target += torch.eye(A_obs_target.shape[3]).cuda()
            A_obs_target = A_obs_target/torch.sum(A_obs_target, dim=3, keepdim=True)


            V_pred, _ = model(V_obs_target, A_obs_target.squeeze())


            V_pred_gt = V_pred_gt.squeeze()
            A_pred_gt = A_pred_gt.squeeze()
            V_pred = V_pred.squeeze()

            if num_iter % batch_size!=0 and i_ != turn_point:
                if first_loss:
                    loss = loss_fn(V_pred, V_pred_gt) 
                    first_loss = False
                else:
                    loss += loss_fn(V_pred, V_pred_gt) 
            
            else:
                loss = loss / batch_size
                first_loss=True

                loss_batch += loss.item()
                print('VAL:','\t Epoch:', epoch,'\t Loss:',loss_batch/num_iter)


    return loss_batch/num_iter