#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from common_shapenet import Config, ConfigRandLA
from models.ffb6d import FFB6D
import datasets.shapenet.shapenet_single_cls as dataset_single_shapenet
from utils.shapenet_eval_utils import cal_frame_poses_shapenet
from utils.basic_utils import Basic_Utils
import matplotlib.pyplot as plt

try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey
import concurrent.futures
import threading

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-dataset", type=str, default="linemod",
    help="Target dataset, ycb or linemod. (linemod as default)."
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
parser.add_argument(
    "-show", action='store_true', help="View from imshow or not."
)

parser.add_argument(
    "-debug", action='store_true', help="Visualize key points or not"
)

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('--gpu', type=str, default="0,1,2,3,4,5,6,7")
args = parser.parse_args()

config = Config(ds_name='shapenet', cls_type=args.cls)
bs_utils = Basic_Utils(config)
if config.test_from_back:
    log_eval_dir = config.log_eval_dir
else:
    log_eval_dir = config.log_eval_dir + '_train'


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint['model_state']
        if 'module' in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec


def cal_view_pred_pose(model, data, epoch=0, obj_id=-1):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys():
            if key == 'obj_name' or key == 'cls_name':
                continue
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32, np.uint16]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()

        labels = cu_dt['labels'][config.n_task * config.test_context:]
        kp_3d = cu_dt['kp_3ds'][config.n_task * config.test_context:,0,:,:]
        #print(kp_3d.shape)
        ctr_3d = cu_dt['ctr_3ds'][config.n_task * config.test_context:,0,:]
        obj_id_lst = data['obj_name'][config.n_task * config.test_context:]
        #print(ctr_3d.shape)

        end_points = model(cu_dt)
        #classes_rgbd = labels # use groud truth
        pcld = end_points['voted_cld']
        classes_rgbd = end_points['pred_rgbd_segs']
        args.debug = True
        args.show = False

        bs = data['obj_name'].shape[0]
        np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[config.n_task * config.test_context:].transpose(0,2,3,1).copy()
        cls_ids = cu_dt['cls_ids'][config.n_task * config.test_context:]
        epoch_lst = [epoch for i in range(bs)]

        data_gen = zip(
            kp_3d,ctr_3d, pcld, classes_rgbd,
            end_points['pred_ctr_ofs'], end_points['pred_kp_ofs'],
            obj_id_lst, np_rgb, cls_ids, epoch_lst
        )
    wk_num = min(8, bs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=wk_num) as executor:
        executor.map(visualise_pose, data_gen)


    vis_dir = os.path.join(config.log_eval_dir, "pose_vis")
    if epoch == 0:
        print("\n\nResults saved in {}".format(vis_dir))

def visualise_pose(data_zip):
    args.debug = True
    args.show = False
    kp_3d,ctr_3d, pcld, classes_rgbd, pred_ctr_ofs,pred_kp_ofs, obj_id, np_rgb, cls_ids,epoch=\
        data_zip
    obj_id = obj_id[0]
    current_thread = threading.current_thread().name
    current_thread_id = current_thread[-1]

    ctr_3d = ctr_3d.view(1, 3)
    kps_3d = torch.cat((kp_3d, ctr_3d), dim=0)

    kp_2ds_gt = bs_utils.project_p3d(kps_3d.cpu().numpy(), 1000.0)
    masks = None # classes_rgbd
    pred_pose_lst, pred_2d_kps = cal_frame_poses_shapenet(
        pcld, masks, pred_ctr_ofs,
        pred_kp_ofs, True, 2, False, obj_id, args.cls,
        debug=True  # for visualization
    )
    pred_cls_ids = np.array([[1]])
    ori_rgb = np_rgb.copy()
    for cls_id in cls_ids.cpu().numpy():
        idx = np.where(pred_cls_ids == cls_id)[0]
        if len(idx) == 0:
            continue
        pose = pred_pose_lst[idx[0]]
        mesh_pts = bs_utils.get_pointxyz(args.cls, obj_id,
                                         ds_type=args.dataset,
                                         mode='sample',
                                         choose_samples=10000).copy()
        mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
        if args.dataset == "shapenet":
            K = config.intrinsic_matrix["shapenet"]
        else:
            raise ValueError('Only shapenet intrinsic_matrix is available.')
        mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
        pcld = pcld.cpu().numpy()
        pcld_mask = bs_utils.project_p3d(pcld, 1.0, K) 
        # color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
        k_iter = config.num_instances_per_item // config.test_mini_batch_size
        color = bs_utils.get_label_color(epoch // k_iter, n_obj=config.max_test_class, mode=2)
        #mask_rgb = bs_utils.draw_p2ds(np_rgb.copy(), pcld_mask, color=color, thickness=2)
        mask_rgb = bs_utils.draw_p2ds_alpha(np_rgb.copy(), pcld_mask, color=color, 
                                      r=2,
                                      thickness=-1,
                                      filled=True, 
                                      alpha=0.1)
        # small alpha -> small transparancy(darker!)
        np_rgb = bs_utils.draw_p2ds_alpha(np_rgb, mesh_p2ds, color=color, 
                                          r=2, 
                                          thickness=-1,
                                          filled=True, 
                                          alpha=0.5)
    
    if config.test_from_back and config.split_vis:
        if obj_id in best_obj_list:
            vis_dir = os.path.join(log_eval_dir, "kp_vis_good")
            vis_dir_2 = os.path.join(log_eval_dir, "pose_vis_good")
            vis_dir_3 = os.path.join(log_eval_dir, "mask_good")
        elif obj_id in worst_obj_list:     
            vis_dir = os.path.join(log_eval_dir, "kp_vis_bad")            
            vis_dir_2 = os.path.join(log_eval_dir, "pose_vis_bad")
            vis_dir_3 = os.path.join(log_eval_dir, "mask_bad")
        else: 
            vis_dir = os.path.join(log_eval_dir, "kp_vis")
            vis_dir_2 = os.path.join(log_eval_dir, "pose_vis")
            vis_dir_3 = os.path.join(log_eval_dir, "mask")
    else:        
        vis_dir = os.path.join(log_eval_dir, "kp_vis")
        vis_dir_2 = os.path.join(log_eval_dir, "pose_vis")
        vis_dir_3 = os.path.join(log_eval_dir, "mask")        
        
    # pose visualization    
    #vis_dir = os.path.join(log_eval_dir, "pose_vis")
    ensure_fd(vis_dir)
    f_pth = os.path.join(vis_dir, "{}_{}_{}.png".format(obj_id, epoch % k_iter, current_thread_id))
    bgr = np_rgb[:, :, ::-1]
    ori_bgr = ori_rgb[:, :, ::-1]
    #cv2.imwrite(f_pth, bgr)

    # mask visualization
    #vis_dir_3 = os.path.join(log_eval_dir, "mask")
    ensure_fd(vis_dir_3)
    f_pth_3 = os.path.join(vis_dir_3, "{}_{}_{}.png".format(obj_id, epoch % k_iter, current_thread_id))
    mask_bgr  = mask_rgb [:, :, ::-1]
    cv2.imwrite(f_pth_3, mask_bgr)

    # visualize pose and keypoints
    #vis_dir_2 = os.path.join(log_eval_dir, "pose_ky")
    ensure_fd(vis_dir_2)
    k_iter = config.num_instances_per_item // config.test_mini_batch_size

    #print('current_thread id = ', current_thread_id)
    f_pth_2 = os.path.join(vis_dir_2, "{}_{}_{}.png".format(obj_id, epoch % k_iter, current_thread_id))
    color_green = (0, 255, 0)  # BGR color
    color_blue = (255, 0, 0)
    color_yellow = (0, 255, 255)
    color_list = [color_green, (49, 99,0), color_yellow,
                    (255, 255, 0), (0, 0, 128), (204, 0, 102), #sky, brawn, purple #(128, 0, 128)
                    (0, 69, 255), (255, 0, 255), color_blue] # orange, pink
    # light yellow: (140, 230, 240)
    # lime: (0, 204, 157)
    # forest (49, 99,0)
    
    # Color_brewer
    #color_list = [(28, 26, 228), (184,126,55), (74,175,77),
    #              (163, 78, 152), (0,127, 225), (51,255,255),
    #              (40,86,166), (191,129, 247), (153 ,153, 153)]

    if pred_2d_kps is not None:
        bgr = bs_utils.draw_p2ds(bgr.astype(np.float32), pred_2d_kps, r=3, color=color_list)
        ori_bgr = bs_utils.draw_p2ds(ori_bgr.astype(np.float32), pred_2d_kps, r=3, color=color_list)
    bgr = bs_utils.draw_p2ds_triangle(bgr.astype(np.float32), kp_2ds_gt, r=4, color=color_list)
    ori_bgr = bs_utils.draw_p2ds_triangle(ori_bgr.astype(np.float32), kp_2ds_gt, r=4, color=color_list)
    cv2.imwrite(f_pth_2, bgr)
    cv2.imwrite(f_pth, ori_bgr)

def main():
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

    if args.dataset == "shapenet":
        test_ds = dataset_single_shapenet.ShapeNetData6D(config.num_instances_per_item,
                                          ds_type='test', categ=args.cls, drop_last=True, DEBUG=False)
    else:
        raise ValueError('Currently demo.py only support ShapeNet.')

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints,training=False
    )
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )
    tqdm_it = tqdm.tqdm(range(test_ds.estimated_it['test']), leave=False, desc='demo')
    for i in tqdm_it:
        data = test_ds.get_batch('test', tasks_per_batch=1)
        if data is None:
            tqdm_it.close()
            print('Eval joint epoch: data loader is empty!')
            break
        obj_id = data['obj_name'][0,0]
        cal_view_pred_pose(model, data, epoch=i, obj_id=obj_id)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
