import torch
from torch import nn
from torch.nn import functional as F
import models.pytorch_utils as pt_utils
from torch.distributions.kl import kl_divergence
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable

# GNN decoder
from torch_geometric.nn import PointConv
from models.gnn_utils import BatchData, BatchPointConv

class Encoder(nn.Module):
    """
    Maps an (x_i, y_i) pair to a representation r_i.
    ----------
    x_dim : Dimension of x values.
    y_dim : Dimension of y values.
    h_dim : Dimension of hidden layer.
    r_dim : Dimension of output representation r.
    """

    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()
        self.x_dim = x_dim # 128
        self.y_dim = y_dim # 4
        self.h_dim = h_dim # 128
        self.r_dim = r_dim # 128

        layers = [nn.Linear(x_dim + y_dim, h_dim), # 132 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim),
                  nn.ReLU(inplace=True)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y, cat_dim=2):
        """
        x : torch.Tensor, shape (bs,M,x_dim) or (bs,M,9,x_dim)
        y : torch.Tensor, shape (bs,M,y_dim) or (bs,M,9,3)
        """
        input_pairs = torch.cat((x, y), dim=cat_dim)
        output = self.input_to_hidden(input_pairs)
        return output


class Aggregator():
    """
    Aggregates representations for every (x_i, y_i) pair into a single
    representation.
    ----------Input----------
    r_i : torch.Tensor, shape (bs, num_points, r_dim)

    ----------Retun----------
    r: shape (bs,1,r_dim)
    """
    def __init__(self, agg_type='max'):
        self.agg_type = agg_type

    def __call__(self,r_i,agg_dim=1):
        if self.agg_type == 'max': # Default type
            r, _ = torch.max(r_i, dim=agg_dim, keepdim=True)
            return r # (3,1,128)

        if self.agg_type == 'mean':
            return torch.mean(r_i, dim=agg_dim,keepdim=True)



class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.
    ----------
    r_dim : Dimension of output representation r.
    z_dim : Dimension of latent variable z.
    """
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor, shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : Dimension of x values.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer.
    y_dim : Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim,n_kps=8):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.n_kps = n_kps

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)

        self.ctr_ofst_layer = (
            pt_utils.Seq(h_dim)
            .conv1d(h_dim, bn=True, activation=nn.ReLU())
            .conv1d(h_dim, bn=True, activation=nn.ReLU())
            .conv1d(h_dim, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        self.kp_ofst_layer = (
            pt_utils.Seq(h_dim)
            .conv1d(h_dim, bn=True, activation=nn.ReLU())
            .conv1d(h_dim, bn=True, activation=nn.ReLU())
            .conv1d(h_dim, bn=True, activation=nn.ReLU())
            .conv1d(self.n_kps*3, activation=None)
        )

        self.hidden_to_kps = nn.Linear(h_dim, self.n_kps * 3)
        self.hidden_to_ctr = nn.Linear(h_dim, 1 * 3)

    def forward(self, x, z):
        """
        x : shape (bs', num_points, x_dim),
        bs' = n_task * mini_batch_size if config.c_in_t is True
        z : shape (bs, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.repeat(1, num_points, 1) # 3*M*128
        z = torch.tile(z, (batch_size//z.shape[0],1,1)) # 6*M*128

        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)

        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs) # (6,M,256)

        pred_kp_ofs = self.hidden_to_kps(hidden)
        pred_ctr_ofs = self.hidden_to_ctr(hidden)

        pred_kp_ofs = pred_kp_ofs.view(batch_size, num_points, self.n_kps, 3)
        pred_ctr_ofs = pred_ctr_ofs.view(batch_size, num_points, 1, 3)

        return pred_kp_ofs, pred_ctr_ofs


class Decoder_local(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : Dimension of x values.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer.
    y_dim : Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim,n_kps=8):
        super(Decoder_local, self).__init__()

        self.x_dim = x_dim # 128
        self.z_dim = z_dim # 128
        self.h_dim = h_dim # 128
        self.y_dim = y_dim # 3
        self.n_kps = n_kps # 8

        layers = [nn.Linear(x_dim + z_dim, h_dim), # 256 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)

        layers = [nn.Linear(h_dim, h_dim//2), # 128 -> 64
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim//2, 3)]

        self.hidden_to_kps = nn.Sequential(*layers) # 128 -> 3
        self.hidden_to_ctr = nn.Sequential(*layers)

    def forward(self, x, z, no_transform=False):
        """
        x : shape (bs', num_points, x_dim),
        bs' = n_task * mini_batch_size if config.c_in_t is True
        z : shape (bs, 1, 9, z_dim)
        Returns
        -------
        """

        batch_size, num_points, x_dim = x.size()
        x = x.view(batch_size, num_points, 1, x_dim)
        x = x.repeat(1, 1, self.n_kps + 1, 1)  # (bs,M,9,128)
        if not no_transform:
            # Repeat z, so it can be concatenated with every x. This changes shape
            # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
            z = z.repeat(1, num_points, 1, 1)  # (3,M,9,128)
            z = torch.tile(z, (batch_size // z.shape[0], 1, 1, 1))  # (6,M,9,128)

        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x, z), dim=3)
        input_pairs_kp = input_pairs[:,:,:self.n_kps,:]
        input_pairs_ctr = input_pairs[:,:,-1,:]

        hidden_kp = self.xz_to_hidden(input_pairs_kp)  # (6,M,8,128)
        hidden_ctr = self.xz_to_hidden(input_pairs_ctr)  # (6,M,1,128)
        pred_kp_ofs = self.hidden_to_kps(hidden_kp)
        pred_ctr_ofs = self.hidden_to_ctr(hidden_ctr)

        pred_ctr_ofs = pred_ctr_ofs.view(batch_size, num_points, 1, 3)

        return pred_kp_ofs, pred_ctr_ofs


class Decoder_global_target(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : Dimension of x values.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer.
    y_dim : Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim,n_kps=8):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_kps = n_kps

        layers = [nn.Linear(x_dim + z_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)

    def forward(self, x, z):
        """
        x : shape (bs', num_points, x_dim),
        bs' = n_task * mini_batch_size if config.c_in_t is True
        z : shape (bs, 1, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (bs', num_points, h_dim).
        """
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.repeat(1, num_points, 1) # 3*M*128
        z = torch.tile(z, (batch_size//z.shape[0],1,1)) # 6*M*128

        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x, z), dim=2)
        hidden = self.xz_to_hidden(input_pairs) # (6,M,128)

        return hidden


class Decoder_local_target(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : Dimension of x values.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer.
    """
    def __init__(self, x_dim, z_dim, h_dim,n_kps=8):
        super().__init__()

        self.x_dim = x_dim # 128
        self.z_dim = z_dim # 128
        self.h_dim = h_dim # 128
        self.n_kps = n_kps # 8

        layers = [nn.Linear(x_dim + z_dim, h_dim), # 256 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)

    def forward(self, x, z):
        """
        x : shape (bs', num_points, x_dim),
        bs' = n_task * mini_batch_size if config.c_in_t is True
        z : shape (bs, 1, 9, z_dim)
        Returns
        -------
        """

        batch_size, num_points, x_dim = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        x = x.view(batch_size, num_points, 1, x_dim)
        x = x.repeat(1, 1, self.n_kps+1, 1) #(bs,M,9,128)
        z = z.repeat(1, num_points, 1, 1)  # (3,M,9,128)
        z = torch.tile(z, (batch_size // z.shape[0], 1, 1, 1))  # (6,M,9,128)

        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x, z), dim=3)
        input_pairs_kp = input_pairs[:,:,:self.n_kps,:]
        input_pairs_ctr = input_pairs[:,:,-1,:]

        hidden_kp = self.xz_to_hidden(input_pairs_kp)  # (6,M,8,128)
        hidden_ctr = self.xz_to_hidden(input_pairs_ctr)  # (6,M,1,128)
        hidden_ctr = hidden_ctr.view(batch_size, num_points, 1, -1)
        hidden = torch.cat((hidden_kp, hidden_ctr), dim=2)

        return hidden # (bs,M,9,128)


class Decoder_global_hidden(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : Dimension of x values.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer.
    y_dim : Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim,n_kps=8):
        super().__init__()

        self.x_dim = x_dim # 128
        self.z_dim = z_dim # 128
        self.h_dim = h_dim # 128
        self.y_dim = y_dim # 3
        self.n_kps = n_kps # 8

        layers = [nn.Linear(x_dim + z_dim, h_dim), # 256 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)

        layers = [nn.Linear(h_dim, h_dim//2), # 128 -> 64
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim//2, 3)]
        self.hidden_to_kps = nn.Sequential(*layers) # 128 -> 3
        self.hidden_to_ctr = nn.Sequential(*layers)


    def forward(self, x, z):
        """
        x : (bs', M, 9, 128)
            bs' = n_task * mini_batch_size if config.c_in_t is True
        z : (bs, 1, 128)
        """

        batch_size, num_points, _, x_dim = x.size() # (bs,M,9,128)
        z = torch.unsqueeze(z,1) # (bs,1,1,128)
        z = z.repeat(1, num_points, 1 + self.n_kps, 1)  # (3,M,9,128)
        z = torch.tile(z, (batch_size // z.shape[0], 1, 1, 1))  # (6,M,9,128)

        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x, z), dim=3)
        input_pairs_kp = input_pairs[:,:,:self.n_kps,:]
        input_pairs_ctr = input_pairs[:,:,-1,:]

        hidden_kp = self.xz_to_hidden(input_pairs_kp)  # (6,M,8,128)
        hidden_ctr = self.xz_to_hidden(input_pairs_ctr)  # (6,M,1,128)
        pred_kp_ofs = self.hidden_to_kps(hidden_kp)
        pred_ctr_ofs = self.hidden_to_ctr(hidden_ctr)

        pred_ctr_ofs = pred_ctr_ofs.view(batch_size, num_points, 1, 3)

        return pred_kp_ofs, pred_ctr_ofs


class Decoder_all(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : Dimension of x values.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer.
    y_dim : Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim,n_kps=8):
        super().__init__()

        self.x_dim = x_dim # 128
        self.z_dim = z_dim # 128
        self.h_dim = h_dim # 128
        self.y_dim = y_dim # 3
        self.n_kps = n_kps # 8

        layers = [nn.Linear(x_dim * 3, h_dim * 2), # 384 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim * 2, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)

        layers = [nn.Linear(h_dim, h_dim//2), # 128 -> 64
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim//2, 3)]
        self.hidden_to_kps = nn.Sequential(*layers) # 128 -> 3
        self.hidden_to_ctr = nn.Sequential(*layers)

    def forward(self, x, z_g, z_l):
        """
        x : (bs', M, 128)
            bs' = n_task * mini_batch_size if config.c_in_t is True
        z_g : (bs, 1, 128)
        z_l : (bs, 1, 9, 128)
        """

        batch_size, num_points, x_dim = x.size() # (bs,M,128)
        x = x.view(batch_size, num_points, 1, x_dim)
        x = x.repeat(1, 1, self.n_kps+1, 1) #(bs,M,9,128)
        z_g = torch.unsqueeze(z_g,1) # (bs,1,1,128)
        z_g = z_g.repeat(1, num_points, 1 + self.n_kps, 1)  # (3,M,9,128)
        z_g = torch.tile(z_g, (batch_size // z_g.shape[0], 1, 1, 1))  # (6,M,9,128)
        z_l = z_l.repeat(1, num_points, 1, 1)  # (3,M,9,128)
        z_l = torch.tile(z_l, (batch_size // z_l.shape[0], 1, 1, 1))  # (6,M,9,128)

        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x, z_g, z_l), dim=3) # (bs,M,9,384)
        input_pairs_kp = input_pairs[:,:,:self.n_kps,:]
        input_pairs_ctr = input_pairs[:,:,-1,:]

        hidden_kp = self.xz_to_hidden(input_pairs_kp)  # (6,M,8,128)
        hidden_ctr = self.xz_to_hidden(input_pairs_ctr)  # (6,M,1,128)
        pred_kp_ofs = self.hidden_to_kps(hidden_kp)
        pred_ctr_ofs = self.hidden_to_ctr(hidden_ctr)

        pred_ctr_ofs = pred_ctr_ofs.view(batch_size, num_points, 1, 3)

        return pred_kp_ofs, pred_ctr_ofs

class Decoder_gnn(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : Dimension of x values.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer.
    y_dim : Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim, n_kps=8):
        super().__init__()

        self.x_dim = x_dim # 128
        self.z_dim = z_dim  # 128
        self.h_dim = h_dim # 128
        self.y_dim = y_dim # 3
        self.n_kps = n_kps # 8

        layers = [nn.Linear(x_dim + z_dim + y_dim, h_dim), # 256+3 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim), # 128 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim), # 128 -> 128
                  nn.ReLU(inplace=True)]
        local_nn = nn.Sequential(*layers)

        layers = [nn.Linear(h_dim, h_dim), # 128 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim), # 128 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, 3)]
        global_nn = nn.Sequential(*layers)
        self.pointnet_conv = BatchPointConv(local_nn=local_nn, global_nn=global_nn,
                                            add_self_loops=False,node_dim=-2)
    
    def forward(self, x, z, full_graph,no_transform=False):
        """
        x : shape (bs', num_points, x_dim),
        bs' = n_task * mini_batch_size if config.c_in_t is True
        z : shape (bs, 1, 9, z_dim)
        edge_index_list: (bs,), (2, num_edge)
        pos_list: (bs, 9, 3)
        Returns
        -------
        """

        batch_size, num_points, x_dim = x.size()

        x = x.view(batch_size, num_points, 1, x_dim)
        x = x.repeat(1, 1, self.n_kps + 1, 1)  # (bs,M,9,128)
        if not no_transform:
            # Repeat z, so it can be concatenated with every x. This changes shape
            # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
            z = z.repeat(1, num_points, 1, 1)  # (3,M,9,128)
            z = torch.tile(z, (batch_size // z.shape[0], 1, 1, 1))  # (6,M,9,128)


        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x, z), dim=3) # (6,M,9,256)

        input_pairs = input_pairs.permute(1,0,2,3).contiguous().view(
            num_points, -1,  input_pairs.shape[-1])
        full_graph.pos = full_graph.pos.repeat(num_points, 1, 1) # (M, 9 * bs, 3)
        full_graph.x = input_pairs # (M, 9 * bs, 256)
        output = self.pointnet_conv(full_graph.x, full_graph.pos, full_graph.edge_index)
        output = output.view(num_points, batch_size, self.n_kps + 1,  3).permute(1,0,2,3)
        pred_kp_ofs = output[:,:,:self.n_kps,:]
        pred_ctr_ofs = output[:,:,self.n_kps:,:]

        return pred_kp_ofs, pred_ctr_ofs
    

class Decoder_segmentation(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : Dimension of x values.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer.
    y_dim : Dimension of y values.
    """
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super().__init__()

        self.x_dim = x_dim # 128
        self.z_dim = z_dim  # 128
        self.h_dim = h_dim # 128
        self.y_dim = y_dim # 2

        layers = [nn.Linear(x_dim + z_dim, h_dim), # 256 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim), # 128 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim), # 128 -> 128
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, y_dim)]        
        self.hidden_to_label = nn.Sequential(*layers)
        
    def forward(self, x, z):
        """
        x : shape (bs', num_points, x_dim),
        z : shape (bs, 1, z_dim)
        Returns: (bs', num_points, 2)
        -------
        """

        batch_size, num_points, x_dim = x.size()
        #print('target shape: ', x.shape)
        
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.repeat(1, num_points, 1)  # (3, M, 128)
        z = torch.tile(z, (batch_size // z.shape[0], 1, 1))  # (6, M, 128)

        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x, z), dim=2) # (6, M, 256)        
        output = self.hidden_to_label(input_pairs) # (bs, M, 2)
        
        return output


class FocalLoss():
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        input = input.contiguous().view(-1,2)
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def L2_loss(x, y, sigma=1.0, normalize=True, reduce=False):
    """
    x: (bs,9,M,3)
    y: (bs,9,M,3)
    """
    bs, n_kpts, n_pts, c = x.size()
    diff = x - y
    abs_diff = torch.abs(diff)
    in_loss = abs_diff
    if normalize:
        l2_error = torch.norm(in_loss, dim=3)
        in_loss = torch.sum(l2_error, 2) / n_pts  # [bs,9]
    if reduce:
        in_loss = torch.mean(in_loss)
    in_loss = in_loss.sum() / bs  # [1] mean batch error
    return in_loss


def L1_focal_loss(x, y, mask):
    """
    x: (bs,9,M,3)
    y: (bs,9,M,3)
    mask: (bs,M)
    """
    bs, n_kpts, n_pts, c = x.size()
    mask = mask.view(-1)
    bg_mask = mask == 0 #(bs * M)
    obj_mask = mask == 1  # (bs * M)

    diff = x - y
    in_loss = torch.abs(diff)
    in_loss = in_loss.permute(1,0,2,3) #(9,bs,M,3)
    in_loss = in_loss.contiguous().view(n_kpts,-1,c) #(9, bs*M, 3)
    bg_loss, obj_loss = in_loss[:,bg_mask,:], in_loss[:, obj_mask, :]
    M_bg, M_obj = bg_loss.shape[1], obj_loss.shape[1]

    bg_loss = torch.sum(bg_loss.view(n_kpts, -1), 1) /(M_bg * c)
    obj_loss = torch.sum(obj_loss.view(n_kpts, -1), 1) / (M_obj * c)
    bg_loss = bg_loss.sum()
    obj_loss = obj_loss.sum()
    if M_obj == 0:
        print('!! No pixel is on the object !!')
        if bg_loss >= 1:
            obj_loss = bg_loss
        else:
            obj_loss  = bg_loss * torch.log10(bg_loss) * (-1) * 5.0
    return bg_loss, obj_loss

def L1_loss(x, y, sigma=1.0, normalize=True, reduce=False):
    """
    x: (bs,9,M,3)
    y: (bs,9,M,3)
    """
    bs, n_kpts, n_pts, c = x.size()
    diff = x - y
    abs_diff = torch.abs(diff)
    in_loss = abs_diff
    if normalize:
        in_loss = torch.sum(
            in_loss.view(bs, n_kpts, -1), 2
        ) / (c * n_pts)  # [bs,9]
    if reduce:
        in_loss = torch.mean(in_loss)
    in_loss = in_loss.sum() / bs  # [1] mean batch error
    return in_loss

def kl_loss(q_target, q_context):
    kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
    return kl

def global_similarity_loss(x, y):
    """
    x: (bs,9,M,3)
    y: (bs,9,M,3)
    """
    x_dist = get_nomrmalized_dist_matrix(x)
    y_dist = get_nomrmalized_dist_matrix(y)
    loss = torch.linalg.norm(x_dist-y_dist,dim=(-2,-1))
    loss = torch.mean(loss)
    return loss.data

def get_nomrmalized_dist_matrix(batch_points):
    bs, n_kp, num_points, c = batch_points.shape
    batch_points = batch_points.permute(0, 2, 1, 3)
    temp_points_1 = batch_points.view(bs, num_points, n_kp, 1, c).repeat(1, 1, 1, n_kp, c)
    temp_points_2 = batch_points.view(bs, num_points, 1, n_kp, c).repeat(1, 1, n_kp, 1, c)
    dis = torch.norm(temp_points_1 - temp_points_2, dim=4)
    di_scaled = min_max_norm(dis)  # (bs, num_points, 8, 8)
    return di_scaled

def min_max_norm(v):
    bs, num_points, n, _ = v.shape
    v = v.view(bs, num_points, -1)
    unique_values = torch.unique(v, dim=-1, sorted=True)
    v_max = unique_values[:, :, -1].view(bs, num_points, 1)
    v_min = unique_values[:, :, 1].view(bs, num_points, 1)
    v_scaled = (v - v_min) / (v_max - v_min)
    v_scaled = v_scaled.view(bs, num_points, n, n)
    mask = torch.eye(n, n).byte()
    mask = mask.view(1, 1, n, n).repeat(bs, num_points, 1, 1).cuda()
    v_scaled.masked_fill_(mask, 0.0)
    return v_scaled

class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        torch.nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x

def main():
    a = torch.tensor([[0.0, 0], [2, 0], [0, 8],[5,5]])
    b = a.view(1,1, 4, -1)
    b = b.repeat(1,4, 1, 1)
    print(b.shape)
    c = b.clone()
    b = b.permute(0, 2, 1, 3).cuda()
    c = c.permute(0,2, 1, 3).cuda()
    loss = global_similarity_loss(b,c)
    print('Loss is: ', loss.item())

if __name__ == "__main__":
    main()