import torch 
import numpy as np

def bivariate_loss(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result

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

        _, _, _, _, _, _, V_obs_source, A_obs_source, V_pred_gt, A_pred_gt = batch_source
        _, _, _, _, _, _, V_obs_target, A_obs_target, _, _ = batch_target

        A_obs_source += torch.eye(A_obs_source.shape[3]).cuda()
        A_obs_source = A_obs_source/torch.sum(A_obs_source, dim=3, keepdim=True)

        A_obs_target += torch.eye(A_obs_target.shape[3]).cuda()
        A_obs_target = A_obs_target/torch.sum(A_obs_target, dim=3, keepdim=True)

        optimizer.zero_grad()

        V_pred, _, L_align = model(V_obs_source, A_obs_source.squeeze(), V_obs_target, A_obs_target.squeeze())

        V_pred = V_pred.permute(0,2,3,1)

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

    for i_, batch_target in enumerate(target_loader):
        num_iter += 1

        batch_target = [tensor.cuda() for tensor in batch_target]

        _, _, _, _, _, _, V_obs_target, A_obs_target, V_pred_gt, A_pred_gt = batch_target

        A_obs_target += torch.eye(A_obs_target.shape[3]).cuda()
        A_obs_target = A_obs_target/torch.sum(A_obs_target, dim=3, keepdim=True)


        V_pred, _ = model(V_obs_target, A_obs_target.squeeze())

        V_pred = V_pred.permute(0,2,3,1)

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