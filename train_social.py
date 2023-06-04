
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

from mydataset import * 

import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from mymodel import *

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
parser.add_argument('--dataset', default='data_test/raw/eth',
                    help='eth,hotel,univ,zara1,zara2')    

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=250,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.01,
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


dset_train = TrajectoryDataset(
        data_set+'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

loader_train = DataLoader(
        dset_train,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=0)


dset_val = TrajectoryDataset(
        data_set+'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

loader_val = DataLoader(
        dset_val,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =False,
        num_workers=1)


#Defining the model 

model = T_GNN(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()


#Training settings 

optimizer = optim.SGD(model.parameters(),lr=args.lr)

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
    global metrics,loader_train
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1


    for cnt,batch in enumerate(loader_train): 
        batch_count+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        V_obs,A_obs,V_pred_gt,A_pred_gt = batch

        A_obs += torch.eye(A_obs.shape[3]).cuda()
        A_obs = A_obs/torch.sum(A_obs, dim=3, keepdim=True)

        optimizer.zero_grad()

        V_pred,_ = model(V_obs,A_obs.squeeze())

        V_pred = V_pred.permute(0,2,3,1)



        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

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
    if args.use_lrschd:
        scheduler.step()


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