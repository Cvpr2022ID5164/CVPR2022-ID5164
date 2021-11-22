from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import time
import tqdm
import copy
import shutil
import argparse
import resource
import numpy as np
import cv2
import pickle as pkl
from collections import namedtuple
from cv2 import imshow, waitKey
import random
random.seed(7)

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
import pandas as pd

import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
#import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR, StepLR
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from common_shapenet import Config, ConfigRandLA
import models.pytorch_utils as pt_utils
from models.ffb6d import FFB6D
#from models.loss import OFLoss, FocalLoss
from utils.shapenet_eval_utils import TorchEval, cal_shapenet_add
from utils.basic_utils import Basic_Utils
import datasets.shapenet.shapenet_single_cls as dataset_single_shapenet
import datasets.shapenet.shapenet_multi_cls as dataset_multi_shapenet
from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model
from apex import amp
from apex.multi_tensor_apply import multi_tensor_applier

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-weight_decay", type=float, default=0,
    help="L2 regularization coeff [default: 0.0]",
)
parser.add_argument(
    "-lr", type=float, default=1e-2,
    help="Initial learning rate [default: 1e-2]"
)
parser.add_argument(
    "-lr_decay", type=float, default=0.5,
    help="Learning rate decay gamma [default: 0.5]",
)
parser.add_argument(
    "-decay_step", type=float, default=2e5,
    help="Learning rate decay step [default: 20]",
)
parser.add_argument(
    "-bn_momentum", type=float, default=0.9,
    help="Initial batch norm momentum [default: 0.9]",
)
parser.add_argument(
    "-bn_decay", type=float, default=0.5,
    help="Batch norm momentum decay gamma [default: 0.5]",
)
parser.add_argument(
    "-checkpoint", type=str, default=None,
    help="Checkpoint to start from"
)
parser.add_argument(
    "-epochs", type=int, default=1000, help="Number of epochs to train for"
)
parser.add_argument(
    "-eval_net", action='store_true', help="whether is to eval net."
)
parser.add_argument(
    '--cls', type=str, default="ape",
    help="Target object. (ape, benchvise, cam, can, cat, driller," +
    "duck, eggbox, glue, holepuncher, iron, lamp, phone, hb01-hb33)"
)
parser.add_argument(
    '--test_occ', action="store_true", help="To eval occlusion linemod or not."
)
parser.add_argument("-test", action="store_true")
parser.add_argument("-test_pose", action="store_true")
parser.add_argument("-test_gt", action="store_true")
parser.add_argument("-cal_metrics", action="store_true")
parser.add_argument("-view_dpt", action="store_true")
parser.add_argument('-debug', action='store_true')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--gpu_id', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=8, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parser.add_argument('--epochs', default=2, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--gpu', type=str, default="0,1,2,3,4,5,6,7")
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--keep_batchnorm_fp32', default=True)
parser.add_argument('--opt_level', default="O0", type=str,
                    help='opt level of apex mix presision trainig.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = Config(ds_name='shapenet', cls_type=args.cls)
bs_utils = Basic_Utils(config)
writer = SummaryWriter(log_dir=config.log_traininfo_dir)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[0], rlimit[1]))

color_lst = [(0, 0, 0)]
for i in range(config.n_objects):
    col_mul = (255 * 255 * 255) // (i+1)
    color = (col_mul//(255*255), (col_mul//255) % 255, col_mul % 255)
    color_lst.append(color)


lr_clip = 1e-5
bnm_clip = 1e-2


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel) or \
                isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
        "amp": amp.state_dict(),
    }

def save_checkpoint(
        state, is_best, filename="checkpoint", bestname="model_best",
        bestname_pure='ffb6d_best'
):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{}.pth.tar".format(bestname))
        shutil.copyfile(filename, "{}.pth.tar".format(bestname_pure))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        ck = torch.load(filename,map_location=torch.device(args.local_rank))
        epoch = ck.get("epoch", 0)
        it = ck.get("it", 0.0)
        best_prec = ck.get("best_prec", None)
        if model is not None and ck["model_state"] is not None:
            ck_st = ck['model_state']
            if 'module' in list(ck_st.keys())[0]:
                tmp_ck_st = {}
                for k, v in ck_st.items():
                    tmp_ck_st[k.replace("module.", "")] = v
                ck_st = tmp_ck_st
            model.load_state_dict(ck_st)
            torch.distributed.barrier() 
        if optimizer is not None and ck["optimizer_state"] is not None:
            optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("amp", None) is not None:
            amp.load_state_dict(ck["amp"])
        print("==> Done")
        print("Model at epoch {}".format(epoch))
        return it, epoch, best_prec
    else:
        print("==> ck '{}' not found".format(filename))
        return None


def view_labels(rgb_chw, cld_cn, labels, K=config.intrinsic_matrix['shapenet']):

    #rgb_hwc = np.transpose(rgb_chw[0].numpy(), (1, 2, 0)).astype("uint8").copy()
    #cld_nc = np.transpose(cld_cn.numpy(), (1, 0)).copy()
    rgb_hwc = np.transpose(rgb_chw[0], (1, 2, 0)).astype("uint8").copy()
    cld_nc = np.transpose(cld_cn, (1, 0)).copy()

    p2ds = bs_utils.project_p3d(cld_nc, 1.0, K).astype(np.int32)
    labels = labels.squeeze().contiguous().cpu().numpy()
    colors = []
    h, w = rgb_hwc.shape[0], rgb_hwc.shape[1]
    rgb_hwc = np.zeros((h, w, 3), "uint8") + 255
    for i,lb in enumerate(labels):
        if int(lb) == 0:
            #print('---', p2ds[i])
            c = (255, 255, 255)
        else:
            print(p2ds[i])
            c = (0,0,255)#color_lst[int(lb)]
        colors.append(c)
    show = bs_utils.draw_p2ds(rgb_hwc, p2ds, 1, 1,colors)
    return show

class Model_forward():
    def __init__(self, multiple_eval=0):
        if multiple_eval == 0:
            self.teval = TorchEval()
        else:
            for i in range(multiple_eval):  # config.n_new_test_cls
                self.teval[str(i)] = TorchEval()

    def forward(self, model, data, it=0, epoch=0,
                is_eval=False, is_test=False, finish_test=False,
                test_pose=False, cls_id=None,
                #test_id=0
                ):

        if is_eval:
            model.eval()
        with torch.set_grad_enabled(not is_eval):
            cu_dt = {}
            # device = torch.device('cuda:{}'.format(args.local_rank))
            for key in data.keys():
                if key == 'obj_name' or key=='cls_name':
                    continue
                if data[key].dtype in [np.float32, np.uint8]:
                    cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
                elif data[key].dtype in [np.int32, np.uint32, np.uint16]:
                    cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
                elif data[key].dtype in [torch.uint8, torch.float32]:
                    cu_dt[key] = data[key].float().cuda()
                elif data[key].dtype in [torch.int32, torch.int16]:
                    cu_dt[key] = data[key].long().cuda()

            end_points = model(cu_dt)
            labels = cu_dt['labels']
            # print('label shape', labels.shape) # 2, 12800

            loss_kp_of = end_points['loss_kp_of_obj']
            loss_ctr_of = end_points['loss_ctr_of_obj']

            kl_loss = end_points['kl_loss']
            loss_rgbd_seg = end_points['loss_rgbd_seg']
            acc_rgbd = end_points['acc_rgbd']
            false_positive = end_points['false_positive']

            if kl_loss is not None and config.use_kl_loss:
                loss_lst = [
                    (loss_rgbd_seg, 2.5), (loss_kp_of, 1.2), (loss_ctr_of, 1.0), (kl_loss, 1.0),
                ]
            else:
                loss_lst = [
                    (loss_rgbd_seg, 2.5), (loss_kp_of, 1.2), (loss_ctr_of, 1.0),
                ]

            loss = sum([ls * w for ls, w in loss_lst])

            #_, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1) # return value,index
            #cls_rgbd = labels  # use segmentataion ground truth
            cls_rgbd = end_points['pred_rgbd_segs']
            if config.use_NP:
                cls_rgbd = None

            if kl_loss is not None and config.use_kl_loss:
                loss_dict = {
                    'loss_rgbd_seg': loss_rgbd_seg.item(), # not using segmentataion ground truth
                    'loss_kp_of_obj': loss_kp_of.item(),
                    'loss_ctr_of_obj': loss_ctr_of.item(),
                    'kl_loss': kl_loss,
                    'loss_all': loss.item(),
                    'loss_target': loss.item()
                }
            else:
                loss_dict = {
                    'loss_rgbd_seg': loss_rgbd_seg.item(), # not using segmentataion ground truth
                    'loss_kp_of_obj': loss_kp_of.item(),
                    'loss_ctr_of_obj': loss_ctr_of.item(),
                    'loss_all': loss.item(),
                    'loss_target': loss.item()
                }

            acc_dict = {
                'acc_rgbd': acc_rgbd.item(),
                'false_positive': false_positive.item()
            }
            info_dict = loss_dict.copy()
            info_dict.update(acc_dict)

            if not is_eval:
                if args.local_rank == 0:
                    writer.add_scalars('loss', loss_dict, it)
                    writer.add_scalars('train_seg_acc', acc_dict, it)

            if is_test and test_pose:
                if config.use_NP:
                    cld = end_points['voted_cld']
                    start_idx = config.n_task * config.test_context
                else:
                    cld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
                    start_idx = 0
                    
                obj_id = data['obj_name'][start_idx, 0]
                cls_name = data['cls_name'][start_idx, 0]
                if not args.test_gt:
                    # eval pose from point cloud prediction.
                    add_list, adds_list = self.teval.eval_pose_parallel(
                        cld, cu_dt['rgb'][start_idx:], cls_rgbd, end_points['pred_ctr_ofs'],
                        cu_dt['ctr_targ_ofst'][start_idx:], labels, epoch, cu_dt['cls_ids'][start_idx:],
                        cu_dt['RTs'][start_idx:], end_points['pred_kp_ofs'],
                        cu_dt['kp_3ds'][start_idx:], cu_dt['ctr_3ds'][start_idx:],
                        ds='shapenet', obj_id=obj_id, cls_name=cls_name,
                        min_cnt=1, use_ctr_clus_flter=True, use_ctr=True,
                    )
                else:
                    # test GT labels, keypoint and center point offset
                    gt_ctr_ofs = cu_dt['ctr_targ_ofst'].unsqueeze(2).permute(0, 2, 1, 3)
                    gt_kp_ofs = cu_dt['kp_targ_ofst'].permute(0, 2, 1, 3)
                    add_list, adds_list = self.teval.eval_pose_parallel(
                        cld, cu_dt['rgb'], labels, gt_ctr_ofs,
                        cu_dt['ctr_targ_ofst'], labels, epoch, cu_dt['cls_ids'],
                        cu_dt['RTs'], gt_kp_ofs,
                        cu_dt['kp_3ds'], cu_dt['ctr_3ds'],
                        ds='shapent', obj_id=obj_id, cls_name=cls_name,
                        min_cnt=1, use_ctr_clus_flter=True, use_ctr=True
                    )

        if is_test and test_pose:
            return (end_points, loss, info_dict, add_list, adds_list)
        else:
            return (end_points, loss, info_dict)

class Trainer(object):
    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    """

    def __init__(
        self,
        model,
        model_fn,
        optimizer,
        checkpoint_name="ckpt",
        best_name="best",
        lr_scheduler=None,
        bnm_scheduler=None,
        viz=None,
    ):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model,
            model_fn,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
        )

        self.checkpoint_name, self.best_name = checkpoint_name, best_name

        self.training_best, self.eval_best = {}, {}
        self.viz = viz
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None
        self.world_size = torch.distributed.get_world_size()
        print('World size / n_gpus = ', self.world_size)

    def eval_model(self, d_loader, is_test=False, test_pose=False, it=0,
                   mode='test', shuffle_obj=False,printOption=False):
        self.model.eval()
        eval_dict = {}
        total_loss = 0.0
        count = 1
        iter_count_per_obj = d_loader.iter_count[mode]
        obj_iter_count = 0
        #tqdm_it = tqdm.tqdm(range(d_loader.estimated_it[mode]) , leave=False, desc=mode)
        #print('Estimated {}  iter = '.format(mode, d_loader.estimated_it[mode]))

        temp_kp_loss = 0
        temp_ctr_loss = 0
        temp_seg_loss = 0
        temp_seg_acc = 0
        temp_fasle_positive = 0
        temp_add_list = []
        temp_adds_list = []
        temp_dict = {}
        output_path = os.path.join(config.torch_eval_dir, args.cls + '.csv')

        for j in range(config.eval_iter):
            print('----------> Start iteration: {} <----------'.format(j+1))
            tqdm_it = tqdm.tqdm(range(d_loader.estimated_it[mode]) , leave=False, desc=mode)
            for i in tqdm_it:
                data = d_loader.get_batch(mode, tasks_per_batch=1)
                if data is None:
                    tqdm_it.close()
                    d_loader.reset_set_epoch(mode, epoch= j + 1,shuffle_obj=shuffle_obj)
                    print('Eval joint epoch: data loader is empty!')
                    break

                if data.get('invalid_it', False):
                    continue

                obj_iter_count += 1
                count += 1
                self.optimizer.zero_grad()

                if is_test and test_pose:
                    _, loss, eval_res, add_list, adds_list = self.model_fn.forward(
                        self.model, data, is_eval=True, is_test=is_test, test_pose=test_pose
                    )
                    temp_add_list.extend(add_list)
                    temp_adds_list.extend(adds_list)
                else:
                    _, loss, eval_res = self.model_fn.forward(
                        self.model, data, is_eval=True, is_test=is_test, test_pose=test_pose
                    )

                if 'loss_target' in eval_res.keys():
                    total_loss += eval_res['loss_target']
                else:
                    total_loss += loss.item()
                for k, v in eval_res.items():
                    if v is not None:
                        eval_dict[k] = eval_dict.get(k, []) + [v]

                temp_kp_loss += eval_res['loss_kp_of_obj']
                temp_ctr_loss += eval_res['loss_ctr_of_obj']
                temp_seg_loss += eval_res['loss_rgbd_seg']
                temp_seg_acc += eval_res['acc_rgbd']
                temp_fasle_positive += eval_res['false_positive']
                
                if obj_iter_count == iter_count_per_obj:
                    obj_iter_count = 0 #reset for next object
                    temp_dict['cls_name'] = data['cls_name'][0,0]
                    temp_dict['obj_name']  = data['obj_name'][0,0]
                    temp_dict['loss_kp_of_obj'] = temp_kp_loss / iter_count_per_obj
                    temp_dict['loss_ctr_of_obj'] = temp_ctr_loss / iter_count_per_obj
                    temp_dict['loss_rgbd_seg'] = temp_seg_loss / iter_count_per_obj
                    temp_dict['seg_acc'] = temp_seg_acc / iter_count_per_obj
                    temp_dict['false_positive'] = temp_fasle_positive / iter_count_per_obj
                    if is_test and test_pose:
                        temp_add_dict = cal_shapenet_add(
                            data['cls_name'][0,0], 
                            data['obj_name'][0,0], 
                            temp_add_list, 
                            adds_list=temp_adds_list, 
                            printOption=False)
                        temp_dict.update(temp_add_dict)

                    df = pd.Series(temp_dict).to_frame().T
                    with open(output_path, 'a') as f:
                        df.to_csv(f, index=False, header=f.tell() == 0)
                    temp_kp_loss = 0
                    temp_ctr_loss = 0
                    temp_seg_loss = 0
                    temp_seg_acc = 0
                    temp_fasle_positive = 0
                    temp_add_list = []
                    temp_adds_list = []
            d_loader.reset_set_epoch(mode, 
                                     epoch= j + 10 + config.numpy_seed,
                                     shuffle_obj=shuffle_obj)
            print('---------->> Finish iteration: {} <<----------'.format(j+1))
            
        mean_eval_dict = {}
        acc_dict = {}
        loss_dict = {}

        for k, v in eval_res.items():
            per = 100 if 'acc' in k else 1
            mean_eval_dict[k] = np.array(v).mean() * per
            if 'acc' in k:
                acc_dict[k] = v
            if 'loss' in k:
                loss_dict[k] = v
        if printOption:
            for k, v in mean_eval_dict.items():
                print(k, v)

        if args.local_rank == 0:
            writer.add_scalars('val_loss', loss_dict, it)
            writer.add_scalars('val_seg_acc', acc_dict, it)

        return total_loss / count, eval_dict


    def eval_joint_epoch(self, d_loader, mode, is_test=False, test_pose=False,
                         it=0, shuffle_obj=False, new_class=False):
        '''
        Args:
            d_loader: Dataset class
            mode: 'test', 'val'
        '''
        self.model.eval()
        eval_dict = {}
        total_loss = 0.0
        count = 1
        tqdm_it = tqdm.tqdm(range(d_loader.estimated_it[mode]) , leave=False, desc=mode)
        #print('Estimated {}  iter = '.format(mode, d_loader.estimated_it[mode]))
        for j in range(config.eval_iter):
            #print('Iter: ', j)
            for i in tqdm_it:
                data = d_loader.get_batch(mode, tasks_per_batch=config.n_task)
                if data is None:
                    tqdm_it.close()
                    d_loader.reset_set_epoch(mode, epoch= j + 1,shuffle_obj=shuffle_obj)
                    print('Eval joint epoch: data loader is empty!')
                    break

                if data.get('invalid_it', False):
                    continue
                    
                count += 1
                self.optimizer.zero_grad()

                _, loss, eval_res = self.model_fn.forward(
                    self.model, data, is_eval=True, is_test=is_test, test_pose=test_pose
                )

                if 'loss_target' in eval_res.keys():
                    total_loss += eval_res['loss_target']
                else:
                    total_loss += loss.item()
                for k, v in eval_res.items():
                    if v is not None:
                        eval_dict[k] = eval_dict.get(k, []) + [v]

        mean_eval_dict = {}
        acc_dict = {}
        loss_dict = {}

        for k, v in eval_res.items():
            per = 100 if 'acc' in k else 1
            mean_eval_dict[k] = np.array(v).mean() * per
            if 'acc' in k:
                acc_dict[k] = v
            if 'loss' in k:
                loss_dict[k] = v

        for k, v in mean_eval_dict.items():
            print(k, v)

        if args.local_rank == 0:
            if new_class:
                writer.add_scalars('new_test_loss', loss_dict, it)
            else:
                writer.add_scalars('val_loss', loss_dict, it)

        return mean_eval_dict['loss_target'], eval_dict

    def train(
        self,
        start_it,
        start_epoch,
        n_epochs,
        train_loader,
        test_loader=None,
        new_class_loader=None,
        best_loss=0.0,
        log_epoch_f=None,
        tot_iter=1,
        clr_div=6,
    ):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : string
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """
        print("Totally train %d iters per gpu." % tot_iter)

        def is_to_eval(epoch, it):
            if it == 1000:
                return True, 1
            wid = tot_iter // clr_div
            if (it // wid) % 2 == 1:
                eval_frequency = wid //15 #15
            else:
                eval_frequency = wid // config.eval_freq_parm
            to_eval = (it % eval_frequency) == 0
            return to_eval, eval_frequency

        def is_to_eval_epoch(epoch, it):
            if it == 1000:
                return True, 1
            if epoch < config.n_total_epoch//2:
                eval_frequency = config.eval_freq_parm
            else:
                eval_frequency = config.eval_freq_parm // 2
            to_eval = (epoch % eval_frequency) == 0
            return to_eval, eval_frequency

        it = start_it
        best_new_class_loss = best_loss
        best_add = -1
        _, eval_frequency = is_to_eval(0, it)
        print('Eval_freq = ', eval_frequency)

        if args.cls=='':
            cls_temp = 'joint'
        else:
            cls_temp = args.cls

        with tqdm.tqdm(range(config.n_total_epoch), desc="%s_epochs" % cls_temp) as tbar, tqdm.tqdm(
            total=eval_frequency, leave=False, desc="train"
        ) as pbar:

            for epoch in tbar:
                if epoch > config.n_total_epoch:
                    break

                if log_epoch_f is not None:
                    os.system("echo {} > {}".format(epoch, log_epoch_f))

                #for batch in train_loader:
                while True:
                    batch = train_loader.get_batch(source='train', tasks_per_batch=config.n_task)
                    if batch is None:
                        train_loader.reset_set_epoch('train', epoch + 1, shuffle_obj=True)
                        print('Finish epoch {}, it = {}'.format(epoch, it))
                        break

                    if batch.get('invalid_it', False):
                        print(batch['invalid_it'])
                        print('Invalid epoch {}, it = {}'.format(epoch, it))
                        it += 1
                        continue
                    #print('epoch {}, it = {}'.format(epoch, it))
                    self.model.train()

                    self.optimizer.zero_grad()

                    _, loss, res = self.model_fn.forward(self.model, batch, it=it)

                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    lr = get_lr(self.optimizer)
                    if args.local_rank == 0:
                        writer.add_scalar('lr/lr', lr, it)

                    self.optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step(it)

                    if self.bnm_scheduler is not None:
                        self.bnm_scheduler.step(it)

                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if self.viz is not None:
                        self.viz.update("train", it, res)

                    eval_flag, eval_frequency = is_to_eval(epoch, it)
                    if eval_flag:
                        pbar.close()
                        local_rank_id = torch.distributed.get_rank()
                        if test_loader is not None:
                            #val_loss, res = self.eval_epoch(test_loader, it=it)
                            train_loader.reset_set_epoch('val',epoch=0,shuffle_obj=False)
                            val_loss, res = self.eval_joint_epoch(train_loader, mode=test_loader,
                                                                  it=it, shuffle_obj=False,
                                                                  new_class=False)
                            print("Epoch{}: it={}, rank={}, val_loss={}".format(epoch, it,local_rank_id, val_loss))
                            val_loss = torch.tensor(val_loss).cuda()
                            torch.distributed.reduce(val_loss,0)

                            if args.local_rank == 0:
                                val_loss = val_loss.item() / self.world_size
                                print('---> Epoch{}: it={}, reduced val_loss = {}'.format(epoch, it, val_loss))
                                is_best = val_loss < best_loss
                                best_loss = min(best_loss, val_loss)

                                save_checkpoint(
                                    checkpoint_state(
                                        self.model, self.optimizer, val_loss, epoch, it
                                    ),
                                    is_best,
                                    filename=self.checkpoint_name,
                                    bestname=self.best_name+'_%.4f' % val_loss,
                                    bestname_pure=self.best_name
                                )
                                info_p = self.checkpoint_name.replace(
                                    '.pth.tar', '_epoch.txt'
                                )
                                os.system(
                                    'echo {} {} >> {}'.format(
                                        it, val_loss, info_p
                                    )
                                )

                        if new_class_loader is not None:
                            train_loader.reset_set_epoch('test', epoch=0, shuffle_obj=False)
                            val_loss, res = self.eval_joint_epoch(train_loader, 
                                                                  mode=new_class_loader,
                                                                  it=it, shuffle_obj=False,
                                                                  new_class=True)
                            print("Epoch{}: it={}, rank={}, val_loss={}".format(epoch, it,
                                                                                local_rank_id,
                                                                                val_loss))
                            val_loss = torch.tensor(val_loss).cuda()
                            torch.distributed.reduce(val_loss,0)
                            
                            if args.local_rank == 0:
                                val_loss = val_loss.item() / self.world_size
                                print('---> Epoch{}: it={}, reduced new_test_loss = {}'.format(epoch, it, val_loss))
                                is_best = val_loss < best_new_class_loss
                                best_new_class_loss = min(best_new_class_loss, val_loss)

                                save_checkpoint(
                                    checkpoint_state(
                                        self.model, self.optimizer, val_loss, epoch, it
                                    ),
                                    is_best,
                                    filename=self.checkpoint_name,
                                    bestname=self.best_name + '_new' + '_%.4f' % val_loss,
                                    bestname_pure=self.best_name + '_new'
                                )
                                info_p = self.checkpoint_name.replace(
                                    '.pth.tar', '_new_epoch.txt'
                                )
                                os.system(
                                    'echo {} {} >> {}'.format(
                                        it, val_loss, info_p
                                    )
                                )

                        pbar = tqdm.tqdm(
                            total=eval_frequency, leave=False, desc="train"
                        )
                        pbar.set_postfix(dict(total_it=it))

            if args.local_rank == 0:
                writer.export_scalars_to_json("./all_scalars.json")
                writer.flush() # make sure that all pending events have been written to disk.
                writer.close()

        return best_loss

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train():
    print("local_rank:", args.local_rank)
    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False   
        cudnn.deterministic = True  
        #torch.use_deterministic_algorithms(True)
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)
        torch.cuda.manual_seed_all(0)    
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    torch.manual_seed(0)     
    np.random.seed(config.numpy_seed + args.local_rank * 10) 
    random.seed(config.numpy_seed + args.local_rank * 10)  # avoid generating the same numbers on different GPUs

    if not args.eval_net: # shapenet single class training
        if args.cls != 'all':
            print('Training ShapeNet class: ', args.cls)
            shapenet_dataset = dataset_single_shapenet.ShapeNetData6D(
                config.num_instances_per_item, ds_type='train', categ=args.cls, drop_last=True, DEBUG=False
            )
        else:
            #train_classes = ['car']
            #test_classes = ['car']
            
            train_classes = ['airplane', 'bag', 'bathtub', 'bed', 'bench', 'bookshelf','bus',
                             'cabinet', 'camera','cap', 'chair', 'earphone', 'motorcycle', 'mug',
                             'table', 'train', 'vessel', 'washer','printer']
            test_classes = ['birdhouse', 'car', 'piano', 'laptop', 'sofa']            
            
            print('Training class: ', train_classes)
            print('Test class: ', test_classes)
            shapenet_dataset = dataset_multi_shapenet.ShapeNetData6D(
                config.num_instances_per_item, ds_type='train',train_categ=train_classes,
                test_categ=test_classes, drop_last=True, DEBUG=False
            )
        train_ds_minibatch_per_epoch = shapenet_dataset.train_batch_per_epoch
    else:
        print('---------->Evaluating ShapeNet class: {} <---------- '.format(args.cls))
        shapenet_dataset_test = dataset_single_shapenet.ShapeNetData6D(
            config.num_instances_per_item, ds_type='test', categ=args.cls, drop_last=True, DEBUG=False
        )

    rndla_cfg = ConfigRandLA
    if args.eval_net:
        model = FFB6D(
            n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
            n_kps=config.n_keypoints,training=False, NP_type= config.np_type
        )
    else:
        model = FFB6D(
            n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
            n_kps=config.n_keypoints,training=True, NP_type= config.np_type
        )

    # for debugging
    #file = open('print.txt','w')
    #print(model,file=file)

    model = convert_syncbn_model(model)
    device = torch.device('cuda:{}'.format(args.local_rank))
    print('local_rank:', args.local_rank)

    model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr_dict.get('start_lr',5*1e-4), weight_decay=args.weight_decay
    )
    opt_level = args.opt_level
    model, optimizer = amp.initialize(
        model, optimizer, opt_level=opt_level,
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1
    
    # load status from checkpoint
    if args.checkpoint is not None:
        if args.eval_net:
            checkpoint_status = load_checkpoint(
                model, None, filename=args.checkpoint[:-8]
            )
        else:
            checkpoint_status = load_checkpoint(
                model, optimizer, filename=args.checkpoint[:-8]
            )
            #torch.distributed.barrier() # the other GPUs should wait for GPU0 to load the model
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
        if args.eval_net:
            assert checkpoint_status is not None, "Failed loadding model."

    if not args.eval_net:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True
        )
        clr_div = 2
        lr_type = config.lr_dict.get('lr_type', 'constant')
        if lr_type == 'constant':
                lr_scheduler = None
        elif lr_type == 'cyclic':   
            base_lr = config.lr_dict.get('start_lr', 1e-5)
            max_lr = config.lr_dict.get('max_lr', 5*1e-4)
            lr_scheduler = CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr,
                cycle_momentum=False,
                step_size_up=config.n_total_epoch * train_ds_minibatch_per_epoch // clr_div // args.gpus,
                step_size_down=config.n_total_epoch * train_ds_minibatch_per_epoch // clr_div // args.gpus,
                mode='triangular'
            )
            print(lr_scheduler.state_dict())
        elif lr_type == 'step':
            step_size = config.n_total_epoch * config.lr_dict.get('step_percentage', 0.9)
            step_size = int(step_size)
            step_size =  step_size * train_ds_minibatch_per_epoch // args.gpus # epoch -> iteration
            gamma = config.lr_dict.get('gamma', 0.5)
            lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            print(lr_scheduler.state_dict())
        if args.checkpoint is not None: # to continue training from checkpoint
            print('Apply new learning rate for checkpoints!')
    else:
        lr_scheduler = None
        
    bnm_lmbd = lambda it: max(
        args.bn_momentum
        * args.bn_decay ** (int(it * config.train_mini_batch_size * config.n_task / args.decay_step)),
        bnm_clip,
    )
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bnm_lmbd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`
    model_fn = Model_forward(multiple_eval=0)

    checkpoint_fd = config.log_model_dir
    if args.cls =='':
        cls_output = 'all'
    else:
        cls_output = args.cls
    trainer = Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name=os.path.join(checkpoint_fd, "FFB6D_%s" % cls_output),
        best_name=os.path.join(checkpoint_fd, "FFB6D_%s_best" % cls_output),
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
    )

    if args.eval_net:
        start = time.time()
        val_loss, res = trainer.eval_model(
            shapenet_dataset_test, is_test=True, test_pose=args.test_pose,
            printOption=False, shuffle_obj=False
        )
        end = time.time()
        print("\nUse time: ", end - start, 's')
    else:
        print('train_ds_minibatch_per_epoch = ', train_ds_minibatch_per_epoch)
        trainer.train(
            it, start_epoch, config.n_total_epoch,
            train_loader = shapenet_dataset,
            test_loader = 'val',
            new_class_loader= 'test', #None, #'test',
            best_loss=best_loss,
            tot_iter=config.n_total_epoch * train_ds_minibatch_per_epoch // args.gpus,
            clr_div=clr_div
        )
        print("Finish training!")
        shapenet_dataset.reset_set_epoch('val', epoch=0, shuffle_obj=False)
        if start_epoch == config.n_total_epoch:
            _ = trainer.eval_joint_epoch(shapenet_dataset, mode='val',shuffle_obj=False)

if __name__ == "__main__":
    args.world_size = args.gpus * args.nodes
    train()
    print("Finish all!")