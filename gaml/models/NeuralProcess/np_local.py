import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import random
import numpy as np
#random.seed(7)
from six.moves import cPickle as pickle
from models.NeuralProcess.np_blocks import Encoder, Aggregator, Decoder_local,Decoder_gnn, Decoder_segmentation
from models.NeuralProcess.np_blocks import L1_loss,L2_loss,AttnLinear,FocalLoss, L1_focal_loss
from common_shapenet import Config
config = Config(ds_name='shapenet')
import argparse
from performer_pytorch import FastAttention
import sys


class NeuralProcess(nn.Module): # NP_local
    """
    Implements Neural Process for functions of arbitrary dimensions.
    Parameters
    ----------
    x_dim : Dimension of x values.
    y_dim : Dimension of y values.
    r_dim : Dimension of output representation r.
    z_dim : Dimension of latent variable z.
    h_dim : Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim,n_kps=8,n_heads=8,training=True):
        super(NeuralProcess, self).__init__()
        self.r_dim = r_dim #128
        self.z_dim = z_dim
        self.h_dim = h_dim # 128
        self.x_dim = x_dim # 128
        self.y_dim = y_dim # 27
        self.training = training
        self.n_kps = n_kps
        self.focal_loss = FocalLoss(gamma=2)

        # Initialize networks
        #self.xy_to_r_g = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.xy_to_r_l = Encoder(x_dim, 4, h_dim, r_dim)
        if config.use_attention:
            self._W_k = nn.ModuleList(
                [AttnLinear(h_dim, h_dim) for _ in range(n_heads)]
            )
            self._W_v = nn.ModuleList(
                [AttnLinear(h_dim*9, h_dim*9) for _ in range(n_heads)]
            )
            self._W_q = nn.ModuleList(
                [AttnLinear(h_dim, h_dim) for _ in range(n_heads)]
            )
            self._W = AttnLinear(n_heads * h_dim * 9, h_dim * 9)
            self.attn = FastAttention(dim_heads=128,
                                 # nb_features=nb_features,
                                 causal=False)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        else:
            self.aggregator = Aggregator(agg_type='max')
        
        # for segmentation
        self.xz_to_seg = Decoder_segmentation(x_dim, z_dim, h_dim, 2)
        
        if config.use_gnn_decoder:
            self.xzl_to_y = Decoder_gnn(x_dim, z_dim, h_dim, 3, n_kps=self.n_kps)
        else:
            self.xzl_to_y = Decoder_local(x_dim, z_dim, h_dim, 3, n_kps=self.n_kps)

    def _multihead_attention(self, k, v, q):
        k_all = []
        v_all = []
        q_all = []

        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)

            k_all.append(k_)
            v_all.append(v_)
            q_all.append(q_)

            #out = self._dot_attention(k_, v_, q_)
            #outs.append(out)
        k_all = torch.stack(k_all, dim=1)
        v_all = torch.stack(v_all, dim=1)
        q_all = torch.stack(q_all, dim=1)
        outs=self.attn(q=q_all, k=k_all, v=v_all)
        outs = outs.permute(0,2,3,1).contiguous()
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep
    
    
    # CNP
    def forward(self, x_context, y_context, x_target, y_target=None,
                label_context=None, label_target=None,
                full_graph=None, voted_cld=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        edege_index_list: (batch_size, ), element in the list: (2, num_edges) # data.edege_index
        kps_ctr_target: (batch_size, 9, 3) # data.pos
        x_label: (bs, Mc)
        y_label: (bs, Mt)

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """

        # Infer quantities from tensor dimensions
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        bs, M, y_dim = y_target.size()

        # Encode target and context
        # Global representation
        #r_g_i = self.xy_to_r_g(x_context, y_context,cat_dim=2)
        # Local representation
        x_context_2 = x_context.view(batch_size, num_context, 1, x_dim)
        x_context_2 = x_context_2.repeat(1, 1, self.n_kps + 1, 1) # (bs,M,9,128)
        y_context_2 = y_context.view(batch_size,num_context, self.n_kps + 1, 3) # (bs,M,9,3)
        label_context_2 = label_context.unsqueeze(-1).unsqueeze(-1)
        label_context_2 = label_context_2.repeat(1, 1, self.n_kps + 1, 1)
        y_context_2 = torch.cat((y_context_2, label_context_2), -1) # (bs,M,9,4)
        r_l_i = self.xy_to_r_l(x_context_2, y_context_2,cat_dim=3) # -> (bs,M,9,128)
        

        # Aggregate r_i into r using max.
        #r_g = self.aggregator(r_g_i) #(bs,1,128)
        if config.use_attention:
            r_l_i = r_l_i.view(batch_size, num_context, -1)
            x_context_2 = torch.tile(x_context, (bs//batch_size,1,1))
            r_l_i = torch.tile(r_l_i, (bs//batch_size,1,1))
            r_l = self._attention_func(x_context_2, r_l_i, x_target)
            r_l = r_l.view(bs, M, self.n_kps +1, self.h_dim) #(bs,M,9,128)
        else:
            r_l = self.aggregator(r_l_i,agg_dim=1) #(bs,1,9,128)
        del x_context_2
        del r_l_i
        
        # predict segmentation
        r_l_seg =  self.aggregator(r_l,agg_dim=2).squeeze(1) #(bs,1,128)
        #print('r_l_seg shape = {}'.format(r_l_seg.shape))
        pred_label = self.xz_to_seg(x_target, r_l_seg) # (bs, M, 2)
        loss_dict = {}
        loss_dict['loss_rgbd_seg'] = self.focal_loss.forward(pred_label, label_target)
        _, cls_rgbd = torch.max(pred_label, 2)  # (bs, M)
        acc_rgbd = (cls_rgbd == label_target).float().sum() / label_target.numel()
        
        wrong_pred =  cls_rgbd != label_target
        gt_bg = label_target == 0
        wrong_pred = torch.logical_and(wrong_pred, gt_bg)
        pred_obj = cls_rgbd == 1
        
        M_obj = []
        
        for i in range(pred_obj.shape[0]):
            M_obj.append(pred_obj[i].nonzero().shape[0])
        #print(M_obj)
        #print('Min M_obj = ', min(M_obj))
        #print('acc_rgbd = ', acc_rgbd)
        M_obj_min = min(min(M_obj), 200)
        #M_obj_min = min(min(M_obj), 50)
        M_obj_min = max(M_obj_min, 30)
        #M_obj_min = max(M_obj_min, 1)
        
        x_target_obj = torch.zeros(bs, M_obj_min, x_target.shape[-1])
        y_target_obj = torch.zeros(bs, M_obj_min, y_target.shape[-1])
        if voted_cld is not None:
            voted_cld_obj = torch.zeros(bs, M_obj_min, voted_cld.shape[-1])
        else: 
            voted_cld_obj = None
        for i in range(x_target.shape[0]):
            all_idx = pred_obj[i].nonzero().view(-1).tolist()
            random.shuffle(all_idx)
            if len(all_idx) < M_obj_min:
                print('Not enough! all_idx shape: ', len(all_idx))
                all_idx = [*range(x_target.shape[1])]
            chosen_idx = random.sample(all_idx, M_obj_min)
            chosen_idx.sort()
            x_target_obj[i] = x_target[i, chosen_idx, :]
            y_target_obj[i] = y_target[i, chosen_idx, :]
            if voted_cld is not None:
                voted_cld_obj[i] = voted_cld[i, chosen_idx, :]
               
        #print('x_target shape = ', x_target_obj.shape)
            
        loss_dict['acc_rgbd'] = acc_rgbd
        if pred_obj.float().sum() ==0:
            print('--------------> No prediction is on the object! <--------------')
            loss_dict['false_positive'] = torch.tensor([-1.0]).cuda()
        else:
            loss_dict['false_positive'] = wrong_pred.float().sum() / pred_obj.float().sum()
            
        # Decoder
        x_target_obj = x_target_obj.cuda()
        if config.use_gnn_decoder:
            pred_kp_ofs, pred_ctr_ofs = self.xzl_to_y(x_target_obj, r_l, full_graph,no_transform=config.use_attention)
            # (bs,M,8,3)  # (bs,M,1,3)  ## (bs,M,1)
        else:
            pred_kp_ofs, pred_ctr_ofs = self.xzl_to_y(x_target_obj, r_l, no_transform=config.use_attention) #(bs,M,8,3)  #(bs,M,1,3)
        pred_kp_ofs = pred_kp_ofs.permute(0,2,1,3).contiguous() #(bs,8,M,3)
        pred_ctr_ofs = pred_ctr_ofs.permute(0,2,1,3).contiguous()
        #y_target_pred= torch.cat((pred_kp_ofs,pred_ctr_ofs),dim=1) #(bs,9,M,3)

        y_target_obj = y_target_obj.cuda()
        y_target = y_target_obj.view(bs, M_obj_min, self.n_kps+1, 3).permute(0, 2, 1, 3).contiguous()  # (bs,9,M,3)
        y_target_kp = y_target[:, :self.n_kps, :, :]
        y_target_ctr = y_target[:, -1, :, :].view(bs, 1, M_obj_min, 3).contiguous()

        if config.loss_type == 'L1':
            loss_dict['loss_kp_of_obj'] = L1_loss(pred_kp_ofs, y_target_kp)
            loss_dict['loss_ctr_of_obj'] = L1_loss(pred_ctr_ofs, y_target_ctr)

        elif config.loss_type == 'L2':
            loss_dict['loss_kp_of'] = L2_loss(pred_kp_ofs, y_target_kp)
            loss_dict['loss_ctr_of'] = L2_loss(pred_ctr_ofs,y_target_ctr)

        return pred_kp_ofs, pred_ctr_ofs, cls_rgbd, loss_dict, voted_cld_obj


def load_dict(filename_):
    with open(filename_,'rb') as f:
        dict = pickle.load(f)
    return dict


def main():
    
    # prase the local_rank argument from command line for the current process
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    device = torch.device('cuda', args.local_rank)
    
    x_dim = 128
    y_dim = 27  # 9x3
    r_dim = 128  # Dimension of representation of context points
    z_dim = 128  # Dimension of sampled latent variable
    h_dim = 128  # Dimension of hidden layers in encoder and decoder
    training = True
    neural_process = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim,
                                        n_kps=8, training=training)

    neural_process = neural_process.to(device)

    inputs = load_dict("./ffb6d_inputs.pkl")

    for k,v in inputs.items():
        inputs[k] = torch.from_numpy(v)

    #torch.save(rgbd_emb, 'rgbd_emb.pt')
    rgbd_emb = torch.load('./rgbd_emb.pt')
    labels = inputs['labels']
    gt_kp_ofst = inputs['kp_targ_ofst']
    gt_ctr_ofst = inputs['ctr_targ_ofst']
    cld = inputs['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()  # (bs,12800,3)

    x_context, y_context, x_target, y_target, voted_cld = neural_process.context_target_split(
        rgbd_emb, gt_kp_ofst, gt_ctr_ofst, labels, max_num=50, cld=cld, c_in_t=False)
    print(device)
    x_context = x_context.cuda()
    y_context = y_context.cuda()
    x_target = x_target.cuda()
    y_target = y_target.cuda()

    #print(neural_process)
    # model parameters: 275254
    print(
        "model parameters:", sum(param.numel() for param in neural_process.parameters())
    )

    pred_kp_ofs, pred_ctr_ofs,loss_dict = \
        neural_process(x_context, y_context, x_target, y_target)


if __name__ == "__main__":

    main()