
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mydataset import * 
from mymodel import *

import pickle
import argparse


parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=3,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=3, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='t_gnn_data/eth',
                    help='eth,hotel,univ,zara1,zara2')    

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')  
parser.add_argument('--lambda_', type=int, default=1,
                    help='hyperparmeter to balance loss terms')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')
                    
args = parser.parse_args()




print('*'*30)
print("Training initiating....")
print(args)


def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = args.dataset+'/'


dset_source = TrajectoryDataset(
        data_set+'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

loader_train = DataLoader(
        dset_source,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=0)


dset_target = TrajectoryDataset(
        data_set+'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

loader_val = DataLoader(
        dset_target,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=1)


#Defining the model 

model = T_GNN(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()


#Training settings 

optimizer = optim.Adam(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    

checkpoint_dir = './checkpoint/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training 
metrics = {'train_loss':[],  'val_loss':[]}
constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}

def train(epoch):
    global metrics,loader_train, loader_val
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1


    for cnt,(batch_source, batch_target) in enumerate(zip(loader_train, loader_val)): 
        batch_count+=1
        #Get data

        batch_source = [tensor.cuda() for tensor in batch_source]
        V_obs_s,A_obs_s,V_pred_gt,A_pred_gt = batch_source

        batch_target = [tensor.cuda() for tensor in batch_target]
        V_obs_t,A_obs_t,_,_ = batch_target

        A_obs_s += torch.eye(A_obs_s.shape[3]).cuda()
        A_obs_s = A_obs_s/torch.sum(A_obs_s, dim=3, keepdim=True)

        A_obs_t += torch.eye(A_obs_t.shape[3]).cuda()
        A_obs_t = A_obs_t/torch.sum(A_obs_t, dim=3, keepdim=True)

        optimizer.zero_grad()

        V_pred, _, L_align = model(V_obs_s,A_obs_s.squeeze(), V_obs_t,A_obs_t.squeeze() )

        V_pred = V_pred.permute(0,2,3,1)



        V_pred_gt = V_pred_gt.squeeze()
        A_pred_gt = A_pred_gt.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_pred_gt)
            if is_fst_loss :
                loss = l + args.lambda_*L_align
                is_fst_loss = False
            else:
                loss += l + args.lambda_*L_align

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)


            optimizer.step()
            #Metrics
            loss_batch += loss.item()
            print('TRAIN:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)

    metrics['train_loss'].append(loss_batch/batch_count)
    


print('Training started ...')
for epoch in range(args.num_epochs):
    train(epoch)
    if epoch == 100:
        optimizer.param_groups[0]['lr'] = 0.0005

    print('*'*30)
    print('Epoch:',args.tag,":", epoch)
    for k,v in metrics.items():
        if len(v)>0:
            print(k,v[-1])


    print(constant_metrics)
    print('*'*30)
    
    with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)
    
    with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)  


torch.save(model.state_dict(),checkpoint_dir+'val_best.pth')  # OK