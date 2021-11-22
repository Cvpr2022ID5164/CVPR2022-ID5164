import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.pspnet import PSPNet
import models.pytorch_utils as pt_utils
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from models.RandLA.RandLANet import Network as RandLANet
#from models.NeuralProcess.np import NeuralProcess
from models.NeuralProcess.data_split import ContextTargetSplit
import importlib
from common_shapenet import Config
config = Config(ds_name='shapenet')

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}
from models.loss import OFLoss

class FFB6D(nn.Module):
    def __init__(
        self, n_classes, n_pts, rndla_cfg, n_kps=8, training=True, NP_type = 'local',config_setting=None
    ):
        super().__init__()

        # ######################## NP settings #########################
        x_dim = 128
        y_dim = 27   # 9x3
        r_dim = 128  # Dimension of representation of context points
        z_dim = 128  # Dimension of sampled latent variable
        h_dim = 128  # Dimension of hidden layers in encoder and decoder
        self.training = training
        self.use_NP = config.use_NP
        if self.use_NP:
            # Import NP class according to the chosen type
            NP_module_name = 'models.NeuralProcess.np_' + NP_type
            print('NP type: ', NP_type)
            print('Use GNN decoder: ', config.use_gnn_decoder)
            NP_module = importlib.import_module(NP_module_name)
            NP_class = getattr(NP_module, 'NeuralProcess')
            self.neural_process = NP_class(x_dim, y_dim, r_dim, z_dim, h_dim,
                                                n_kps=n_kps, training=self.training)
            self.data_split = ContextTargetSplit(n_kps,config_setting)
        else:
            print('-------------> Not use neural processes! Only train FFB6D! <-------------')

        ######################### Loss  #########################
        self.seg_loss = FocalLoss(gamma=2)
        if self.training:
            self.of_loss = OFLoss().cuda()
        else:
            self.of_loss = OFLoss()

        ######################### prepare stages #########################
        self.n_cls = n_classes
        self.n_pts = n_pts
        self.n_kps = n_kps
        cnn = psp_models['resnet34'.lower()]()

        rndla = RandLANet(rndla_cfg)

        self.cnn_pre_stages = nn.Sequential(
            cnn.feats.conv1,  # stride = 2, [bs, c, 240, 320]
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
        )
        self.rndla_pre_stages = rndla.fc0

        # ####################### downsample stages #######################
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,    # stride = 1, [bs, 64, 120, 160]
            cnn.feats.layer2,    # stride = 2, [bs, 128, 60, 80]
            # stride = 1, [bs, 128, 60, 80]
            nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 60, 80]
        ])
        self.ds_sr = [4, 8, 8, 8]

        self.rndla_ds_stages = rndla.dilated_res_blocks

        self.ds_rgb_oc = [64, 128, 512, 1024]
        self.ds_rndla_oc = [item * 2 for item in rndla_cfg.d_out]
        self.ds_fuse_r2p_pre_layers = nn.ModuleList()
        self.ds_fuse_r2p_fuse_layers = nn.ModuleList()
        self.ds_fuse_p2r_pre_layers = nn.ModuleList()
        self.ds_fuse_p2r_fuse_layers = nn.ModuleList()
        for i in range(4):
            self.ds_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i], self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i]*2, self.ds_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

            self.ds_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.ds_rndla_oc[i], self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.ds_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    self.ds_rgb_oc[i]*2, self.ds_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

        # ###################### upsample stages #############################
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
            nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.final),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
        ])
        self.up_rgb_oc = [256, 64, 64]
        self.up_rndla_oc = []
        for j in range(rndla_cfg.num_layers):
            if j < 3:
                self.up_rndla_oc.append(self.ds_rndla_oc[-j-2])
            else:
                self.up_rndla_oc.append(self.ds_rndla_oc[0])

        self.rndla_up_stages = rndla.decoder_blocks

        n_fuse_layer = 3
        self.up_fuse_r2p_pre_layers = nn.ModuleList()
        self.up_fuse_r2p_fuse_layers = nn.ModuleList()
        self.up_fuse_p2r_pre_layers = nn.ModuleList()
        self.up_fuse_p2r_fuse_layers = nn.ModuleList()
        for i in range(n_fuse_layer):
            self.up_fuse_r2p_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i], self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_fuse_r2p_fuse_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i]*2, self.up_rndla_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

            self.up_fuse_p2r_pre_layers.append(
                pt_utils.Conv2d(
                    self.up_rndla_oc[i], self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )
            self.up_fuse_p2r_fuse_layers.append(
                pt_utils.Conv2d(
                    self.up_rgb_oc[i]*2, self.up_rgb_oc[i], kernel_size=(1, 1),
                    bn=True
                )
            )

        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.
        if not self.use_NP:
            self.rgbd_seg_layer = (
                pt_utils.Seq(self.up_rndla_oc[-1] + self.up_rgb_oc[-1])
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(n_classes, activation=None)
            )

            self.ctr_ofst_layer = (
                pt_utils.Seq(self.up_rndla_oc[-1]+self.up_rgb_oc[-1])
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(3, activation=None)
            )

            self.kp_ofst_layer = (
                pt_utils.Seq(self.up_rndla_oc[-1]+self.up_rgb_oc[-1])
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(n_kps*3, activation=None)
            )

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        if len(feature.size()) > 3:
            feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(
            feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

    def _break_up_pc(self, pc):
        xyz = pc[:, :3, :].transpose(1, 2).contiguous()
        features = (
            pc[:, 3:, :].contiguous() if pc.size(1) > 3 else None
        )
        return xyz, features

    def forward(
        self, inputs, end_points=None, scale=1,
    ):
        """
        Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
            dpt_nrm     : FloatTensor [bs, 6, h, w], 3c xyz in meter + 3c normal map
            cld_rgb_nrm : FloatTensor [bs, 9, npts]
            choose      : LongTensor [bs, 1, npts]
            xmap, ymap: [bs, h, w]
            K:          [bs, 3, 3]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################
        if not end_points:
            end_points = {}
        # ResNet pre + layer1 + layer2
        rgb_emb = self.cnn_pre_stages(inputs['rgb'])  # stride = 2, [bs, c, 240, 320]
        # rndla pre
        xyz, p_emb = self._break_up_pc(inputs['cld_rgb_nrm'])
        p_emb = inputs['cld_rgb_nrm']
        p_emb = self.rndla_pre_stages(p_emb)
        p_emb = p_emb.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###################### encoding stages #############################
        ds_emb = []
        for i_ds in range(4):
            # encode rgb downsampled feature
            rgb_emb0 = self.cnn_ds_stages[i_ds](rgb_emb)
            bs, c, hr, wr = rgb_emb0.size()

            # encode point cloud downsampled feature
            f_encoder_i = self.rndla_ds_stages[i_ds](
                p_emb, inputs['cld_xyz%d' % i_ds], inputs['cld_nei_idx%d' % i_ds]
            )
            f_sampled_i = self.random_sample(f_encoder_i, inputs['cld_sub_idx%d' % i_ds])
            p_emb0 = f_sampled_i
            if i_ds == 0:
                ds_emb.append(f_encoder_i)

            # fuse point feauture to rgb feature
            p2r_emb = self.ds_fuse_p2r_pre_layers[i_ds](p_emb0)
            p2r_emb = self.nearest_interpolation(
                p2r_emb, inputs['p2r_ds_nei_idx%d' % i_ds]
            )
            p2r_emb = p2r_emb.view(bs, -1, hr, wr)
            rgb_emb = self.ds_fuse_p2r_fuse_layers[i_ds](
                torch.cat((rgb_emb0, p2r_emb), dim=1)
            )

            # fuse rgb feature to point feature
            r2p_emb = self.random_sample(
                rgb_emb0.reshape(bs, c, hr*wr, 1), inputs['r2p_ds_nei_idx%d' % i_ds]
            ).view(bs, c, -1, 1)
            r2p_emb = self.ds_fuse_r2p_pre_layers[i_ds](r2p_emb)
            p_emb = self.ds_fuse_r2p_fuse_layers[i_ds](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )
            ds_emb.append(p_emb)

        # ###################### decoding stages #############################
        n_up_layers = len(self.rndla_up_stages)
        for i_up in range(n_up_layers-1):
            # decode rgb upsampled feature
            rgb_emb0 = self.cnn_up_stages[i_up](rgb_emb)
            bs, c, hr, wr = rgb_emb0.size()

            # decode point cloud upsampled feature
            f_interp_i = self.nearest_interpolation(
                p_emb, inputs['cld_interp_idx%d' % (n_up_layers-i_up-1)]
            )
            f_decoder_i = self.rndla_up_stages[i_up](
                torch.cat([ds_emb[-i_up - 2], f_interp_i], dim=1)
            )
            p_emb0 = f_decoder_i

            # fuse point feauture to rgb feature
            p2r_emb = self.up_fuse_p2r_pre_layers[i_up](p_emb0)
            p2r_emb = self.nearest_interpolation(
                p2r_emb, inputs['p2r_up_nei_idx%d' % i_up]
            )
            p2r_emb = p2r_emb.view(bs, -1, hr, wr)
            rgb_emb = self.up_fuse_p2r_fuse_layers[i_up](
                torch.cat((rgb_emb0, p2r_emb), dim=1)
            )

            # fuse rgb feature to point feature
            r2p_emb = self.random_sample(
                rgb_emb0.reshape(bs, c, hr*wr), inputs['r2p_up_nei_idx%d' % i_up]
            ).view(bs, c, -1, 1)
            r2p_emb = self.up_fuse_r2p_pre_layers[i_up](r2p_emb)
            p_emb = self.up_fuse_r2p_fuse_layers[i_up](
                torch.cat((p_emb0, r2p_emb), dim=1)
            )

        # final upsample layers:
        rgb_emb = self.cnn_up_stages[n_up_layers-1](rgb_emb)
        f_interp_i = self.nearest_interpolation(
            p_emb, inputs['cld_interp_idx%d' % (0)]
        )
        p_emb = self.rndla_up_stages[n_up_layers-1](
            torch.cat([ds_emb[0], f_interp_i], dim=1)
        ).squeeze(-1)

        bs, di, _, _ = rgb_emb.size()
        rgb_emb_c = rgb_emb.view(bs, di, -1)
        choose_emb = inputs['choose'].repeat(1, di, 1)
        rgb_emb_c = torch.gather(rgb_emb_c, 2, choose_emb).contiguous()
        
        # Use simple concatenation. Good enough for fully fused RGBD feature.
        rgbd_emb = torch.cat([rgb_emb_c, p_emb], dim=1)


        # ###################### prediction stages #############################
        if not self.use_NP:
            
            rgbd_segs = self.rgbd_seg_layer(rgbd_emb) # (bs, 2, 3600)
            pred_kp_ofs = self.kp_ofst_layer(rgbd_emb)
            pred_ctr_ofs = self.ctr_ofst_layer(rgbd_emb)
            
            pred_kp_ofs = pred_kp_ofs.view(
                bs, self.n_kps, 3, -1
            ).permute(0, 1, 3, 2).contiguous()
            pred_ctr_ofs = pred_ctr_ofs.view(
                bs, 1, 3, -1
            ).permute(0, 1, 3, 2).contiguous() #(bs,8,M,3)
            end_points['pred_kp_ofs'] = pred_kp_ofs
            end_points['pred_ctr_ofs'] = pred_ctr_ofs
            labels = inputs['labels'] # (bs, 3600)          
            end_points['loss_kp_of_obj'] = self.of_loss(pred_kp_ofs, inputs['kp_targ_ofst'], labels)
            end_points['loss_ctr_of_obj'] = self.of_loss(pred_ctr_ofs, inputs['ctr_targ_ofst'], labels)
            
            _, cls_rgbd = torch.max(rgbd_segs, 1) # (bs, 3600)
            acc_rgbd = (cls_rgbd == labels).float().sum() / labels.numel()
            
            end_points['loss_rgbd_seg'] = self.seg_loss.forward(rgbd_segs, labels.view(-1))
            end_points['acc_rgbd'] = acc_rgbd
            end_points['pred_rgbd_segs'] = rgbd_segs
            end_points['kl_loss'] = None
            
            wrong_pred =  cls_rgbd != labels
            gt_bg = labels == 0
            wrong_pred = torch.logical_and(wrong_pred, gt_bg)
            pred_obj = cls_rgbd == 1
            if pred_obj.float().sum() ==0:
                print('--------------> No prediction is on the object! <--------------')
                end_points['false_positive'] = torch.tensor([-1.0]).cuda()
            else:
                end_points['false_positive'] = wrong_pred.float().sum() / pred_obj.float().sum()

            return end_points

        ####################### NP stages #############################
        gt_kp_ofst = inputs['kp_targ_ofst'] #(bs,12800,8,3)
        gt_ctr_ofst = inputs['ctr_targ_ofst'] # (bs,12800,3)
        labels = inputs['labels']
        cls_rgbd = None
        cld = inputs['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous() #(bs,12800,3)

        # Prepare data for NP
        voted_cld = None
        kps_ctr = inputs['kps_ctr'] # (bs, 9, 3)
        if config.use_gnn_decoder:
            if self.training:
                x_context, y_context, x_target, y_target, label_context, label_target, full_graph = \
                    self.data_split.context_target_split(rgbd_emb, gt_kp_ofst, gt_ctr_ofst,
                                                         cls_rgbd, labels, kps_ctr,
                                                         max_num=3600, min_num=3600,
                                                         is_training=self.training)
               
            else:
                x_context, y_context, x_target, y_target, label_context, label_target, voted_cld, full_graph = \
                    self.data_split.context_target_split(rgbd_emb, gt_kp_ofst, gt_ctr_ofst,
                                                         cls_rgbd, labels, kps_ctr,
                                                         max_num=config.test_seed_per_image,
                                                         min_num=config.test_seed_per_image,
                                                         cld=cld, is_training=self.training)
                voted_cld = voted_cld.cuda()

            full_graph = full_graph.cuda()
            x_context = x_context.cuda()
            y_context = y_context.cuda()
            x_target = x_target.cuda()
            y_target = y_target.cuda() # (bs,M,27), 27 = (8+1)*3
            label_context = label_context.cuda() #(bs, M)
            label_target = label_target.cuda()

            pred_kp_ofs, pred_ctr_ofs, cls_rgbd, loss_dict,voted_cld_obj = \
                self.neural_process(x_context, y_context, x_target, y_target, label_context, label_target, full_graph, voted_cld)

        else:
            if self.training:
                x_context, y_context, x_target, y_target, label_context, label_target = self.data_split.context_target_split(
                    rgbd_emb, gt_kp_ofst, gt_ctr_ofst, cls_rgbd, labels, max_num=3600, min_num=3600, is_training=self.training)
            else:
                x_context, y_context, x_target, y_target, label_context, label_target, voted_cld = \
                    self.data_split.context_target_split(
                    rgbd_emb, gt_kp_ofst, gt_ctr_ofst, cls_rgbd, labels,
                    max_num=config.test_seed_per_image,
                    min_num=config.test_seed_per_image,
                    cld=cld, is_training=self.training)
                voted_cld = voted_cld.cuda()

            x_context = x_context.cuda()
            y_context = y_context.cuda()
            x_target = x_target.cuda()
            y_target = y_target.cuda() # (bs,M,27), 27 = (8+1)*3
            label_context = label_context.cuda() #(bs, M)
            label_target = label_target.cuda()
            pred_kp_ofs, pred_ctr_ofs, cls_rgbd, loss_dict, voted_cld_obj = \
                self.neural_process(x_context, y_context, x_target, y_target, label_context, label_target,voted_cld=voted_cld)

        end_points['pred_kp_ofs'] = pred_kp_ofs
        end_points['pred_ctr_ofs'] = pred_ctr_ofs
        end_points['pred_rgbd_segs'] = cls_rgbd
        end_points.update(loss_dict)
        end_points['kl_loss'] = None
        if not self.training:
            end_points['voted_cld'] = voted_cld_obj.cuda()

        return end_points



class FocalLoss():
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            # print("fcls input.size", input.size(), target.size())
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
