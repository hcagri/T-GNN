import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class GraphAttentionModule(nn.Module):
    def __init__(self, max_num_peds=100):
        super(GraphAttentionModule, self).__init__()

        self.max_num_peds = max_num_peds
        self.gat_conv = GATConv(max_num_peds, max_num_peds, 4, concat=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, A):  
        # For each time stemp we have an adjacency matrix, shape of A: (T_obs, num_peds, num_peds)
        num_peds = A.size(2)

        # Create edge_index to perform graph attention layer, assume fully connected graph
        adj_matrix = torch.ones(num_peds, num_peds) - torch.eye(num_peds)
        edge_index, _ = dense_to_sparse(adj_matrix.cuda())
        
        new_A = torch.zeros_like(A)

        for idx, A_t in enumerate(A): 
            # each colum vector of A is our feature vector. So make them row vector to feed them into torch geometric
            x = A_t.T
            # Pad them with zeros, since number of pedestrians in each scene differs. 
            x = F.pad(x, (0, self.max_num_peds - num_peds))
            x = self.lrelu(self.gat_conv(x, edge_index))
            new_A[idx,:,:] = x.T[:num_peds, :num_peds] + torch.eye(num_peds).cuda()
        
        return new_A

        
class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format 
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format 
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))  
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = True,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class T_GNN(nn.Module):
    def __init__(self,
                 n_stgcnn=1,
                 n_txpcnn=1,
                 input_feat=2,
                 feat_dim=64,
                 output_feat=5,
                 seq_len=8,
                 pred_seq_len=12,
                 kernel_size=3, 
                 max_num_peds = 100
                 ):
        super(T_GNN,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.lin_proj = nn.Linear(input_feat, feat_dim)
        self.lin_proj_2 = nn.Linear(feat_dim, output_feat)
        self.relu = nn.ReLU()

        self.graph_attn_module = GraphAttentionModule(max_num_peds)
        self.attention_module = adaptive_learning(feat_dim, seq_len)

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(feat_dim,feat_dim,(kernel_size,seq_len)))
        for _ in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(feat_dim,feat_dim,(kernel_size,seq_len)))


        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for _ in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
            
            
        self.prelus = nn.ModuleList()
        for _ in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())


    def forward(self, v_s, a_s, v_t = None, a_t = None):

        # To measure the relative importance of dynamic spatial relations between pedestrians, the graph attention layer 
        # from [67] is adopted here to update the adjacency matrix
        a_s = self.graph_attn_module(a_s)

        ''' Feature Projection '''
        # v: (1, T_obs, num_peds, feat = 2) -> (1, 8, num_peds, 2)
        v_s = self.relu(self.lin_proj(v_s)).permute(0,3,1,2)
        
        if v_t is not None:
            a_t = self.graph_attn_module(a_t)
            v_t = self.relu(self.lin_proj(v_t)).permute(0,3,1,2)


        ''' Spatial-Temporal Feature Extraction '''
        # v: (1, feat = 64, T_obs, num_peds) -> (1, 64, 8, num_peds)
        for k in range(len(self.st_gcns)):
            v_s,a_s = self.st_gcns[k](v_s,a_s)
        if v_t is not None:
            for k in range(len(self.st_gcns)):
                v_t,a_t = self.st_gcns[k](v_t,a_t)


        ''' Temporal Prediction Module '''
        # v: (1, feat = 64, T_obs, num_peds) -> (1, 64, 8, num_peds)
        v = v_s.permute(0, 2, 1, 3).clone()
        a = a_s
        # v: (1, T_obs, feat=64, num_peds) -> (1, 8, 64, num_peds) 
        # The reason for the reshape is, the TCNN module consider the temporal axis as the feature dimension.
        v = self.prelus[0](self.tpcnns[0](v))
        for k in range(1,self.n_txpcnn):
            v =  self.prelus[k](self.tpcnns[k](v)) + v

        # v: (1, T_pred, feat=64, num_peds)  
        v = v.permute(0, 1, 3, 2)
        # v: (1, T_pred, num_peds, feat=64)

        v = self.lin_proj_2(v)
        # v: (1, T_pred, num_peds, feat=5)
        
        if v_t is not None:
            return v, a, v_s, v_t.clone()  # v: (1, out_feat, T_pred, num_peds) -> (1, 5, 12, num_peds), a has the same shape
        return v,a




class adaptive_learning(nn.Module):
    def __init__(self, feat_dim, seq_len) -> None:
        super().__init__()
        self.feat_dim = feat_dim

        self.tanh = nn.Tanh()
        self.W = nn.Linear(feat_dim*seq_len, feat_dim)
        self.h = nn.Linear(feat_dim, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, Fs, Ft):
        ''' 
        Input shapes (1, feat = 64, T_obs, num_peds) turn them into,
            Fs: Features of source domain (1, num_peds, seq_len (T_obs), feat_dim)
            Ft: Features of traget domain (1, num_peds, seq_len (T_obs), feat_dim) '''
        
        Fs = Fs.permute(0, 3, 2, 1)
        Ft = Ft.permute(0, 3, 2, 1)

        Fs = Fs.reshape(Fs.shape[0], Fs.shape[1],  Fs.shape[2] *  Fs.shape[3])
        Ft = Ft.reshape(Fs.shape[0], Ft.shape[1],  Ft.shape[2] *  Ft.shape[3])

        beta_s = self.softmax(self.h(self.tanh(self.W(Fs))))
        beta_t = self.softmax(self.h(self.tanh(self.W(Ft))))

        c_s = torch.einsum('ijh,ijk->k', beta_s, Fs)
        c_t = torch.einsum('ijh,ijk->k', beta_t, Ft)

        L_align = torch.linalg.norm(c_s - c_t) / self.feat_dim

        return L_align


        