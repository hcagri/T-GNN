import torch 
import torch.distributions.multivariate_normal as torchdist

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


def average_displacement_error(pred_traj, pred_traj_gt):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)

    return loss


def final_displacement_error( pred_pos, pred_pos_gt):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=1))

    return loss


def evaluate(model, loader, _config, num_samples=20):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            V_obs, A_obs, V_tr, A_tr = batch

            A_obs += torch.eye(A_obs.shape[3]).cuda()
            A_obs = A_obs/torch.sum(A_obs, dim=3, keepdim=True)

            V_pred, _ = model(V_obs, A_obs.squeeze())

            V_tr = V_tr.squeeze()
            A_tr = A_tr.squeeze()
            V_pred = V_pred.squeeze()

            sx = torch.exp(V_pred[:,:,2]) #sx
            sy = torch.exp(V_pred[:,:,3]) #sy
            corr = torch.tanh(V_pred[:,:,4]) #corr

            cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda() # shape: (12, 3, 2, 2) >> seq, node, feat, feat
            cov[:,:,0,0]= sx*sx
            cov[:,:,0,1]= corr*sx*sy
            cov[:,:,1,0]= corr*sx*sy
            cov[:,:,1,1]= sy*sy
            mean = V_pred[:,:,0:2]

            mvnormal = torchdist.MultivariateNormal(mean,cov)

            ade, fde = [], []
            total_traj += V_tr.size(1)

            for _ in range(num_samples):

                V_pred = mvnormal.sample()

                ade.append(average_displacement_error(
                    V_pred, V_tr
                ))
                fde.append(final_displacement_error(
                    V_pred[-1], V_tr[-1]
                ))
            
            best_ade = min(torch.stack(ade, dim=1).sum(dim=0))
            best_fde = min(torch.stack(fde, dim=1).sum(dim=0))

            ade_outer.append(best_ade)
            fde_outer.append(best_fde)
        ade = sum(ade_outer) / (total_traj * _config['data']['seq_len_pred'])
        fde = sum(fde_outer) / (total_traj)
        return ade, fde