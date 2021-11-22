#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
import pickle as pkl
import concurrent.futures

from common_shapenet import Config
from utils.basic_utils import Basic_Utils
from utils.meanshift_pytorch import MeanShiftTorch
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except Exception:
    from cv2 import imshow, waitKey

try:
    config = Config(ds_name="shapenet")
    bs_utils = Basic_Utils(config)
except Exception as ex:
    print(ex)


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T

def cal_shapenet_add(cls_name, obj_name, add_list, adds_list=None, printOption=False):
    add_auc = bs_utils.cal_auc(add_list)
    diameter = bs_utils.get_diameter(cls_name, obj_name, ds_type='shapenet')
    d_1 = diameter * 0.1
    add_1 = np.mean(np.array(add_list) < d_1) * 100
    
    # ADD < 0.3d
    d_percent = 0.3
    d_3 = diameter * d_percent
    add_3 = np.mean(np.array(add_list) < d_3) * 100
    
    if adds_list is not None:
        adds_1 = np.mean(np.array(adds_list) < d_1) * 100
        adds_3 = np.mean(np.array(adds_list) < d_3) * 100
        

    if printOption:
        print(cls_name, obj_name)
        print("obj_id: ", obj_name, "{} diameter: {} (m)".format(0.1, d_1))
        print("obj_id: ", obj_name, "{} diameter: {} (m)".format(d_percent, d_3))
        print("***************add auc:\t", add_auc)
        #print("***************adds auc:\t", adds_auc)
        #print("***************add(-s) auc:\t", add_s_auc)
        print("***************add < {} diameter:\t{}".format(0.1, add_1))
        #print("***************adds < {} diameter:\t{}".format(0.1, adds_1))
        print("***************add < {} diameter:\t{}".format(d_percent, add_3))
        #print("***************adds < {} diameter:\t{}".format(d_percent, adds))
    if adds_list is not None:
        add_dict = dict(
            add_1=add_1,
            add_3=add_3,
            adds_1=adds_1,
            adds_3=adds_3,
        )
    else:
        add_dict = dict(
            add_1=add_1,
            add_3=add_3
        )        
    
    return add_dict




# ############################## ShapeNet Evaluation ##############################

def cal_frame_poses_shapenet(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter, obj_id,cls_name,
    debug=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    n_kps, n_pts, _ = pred_kp_of.size() # n_pts = M
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of
    if mask is not None:
        cls_id = 1
        cls_msk = mask == cls_id
        cls_msk = cls_msk.view(-1)
        
    radius = 0.08
    if use_ctr:
        #cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
        cls_kps = torch.zeros(n_cls, n_kps+1, 3)
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id = 1

    kp_2ds = None

    if n_pts < 1 or (mask is not None and cls_msk.sum() < 1):
            pred_pose_lst.append(np.identity(4)[:3, :])
    else:
        if mask is None:
            cls_voted_kps = pred_kp
            ms = MeanShiftTorch(bandwidth=radius)
            ctr, ctr_labels = ms.fit(pred_ctr)
        else:
            cls_voted_kps = pred_kp[:, cls_msk, :]
            ms = MeanShiftTorch(bandwidth=radius)
            ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])

        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr            
        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)
            
        # distinguish Linemod and new dataset
        ds_type = 'shapenet'
        # visualize
        '''
        if debug:
            show_kp_img = np.zeros((480, 640, 3), np.uint8)
            kp_2ds = bs_utils.project_p3d(
                cls_kps[cls_id].cpu().numpy(), 1000.0, K=ds_type
            )
            # print("cls_id = ", cls_id)
            # print("kp3d:", cls_kps[cls_id])
            # print("kp2d:", kp_2ds, "\n")
            color = (0, 0, 255)  # bs_utils.get_label_color(cls_id.item())
            show_kp_img = bs_utils.draw_p2ds(show_kp_img, kp_2ds, r=3, color=color)
            imshow("kp: cls_id=%d" % cls_id, show_kp_img)
            waitKey(0)
        '''

        kp_2ds = bs_utils.project_p3d(cls_kps[cls_id].cpu().numpy(), 1000.0) #K=ds_type
        mesh_kps = bs_utils.get_kps(cls_name, obj_id, ds_type=ds_type)
        if use_ctr:
            mesh_ctr = bs_utils.get_ctr(cls_name, obj_id, ds_type=ds_type).reshape(1, 3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        # mesh_kps = torch.from_numpy(mesh_kps.astype(np.float32)).cuda()
        pred_RT = best_fit_transform(
            mesh_kps,
            cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        )
        pred_pose_lst.append(pred_RT)

    if debug:
        return (pred_pose_lst, kp_2ds)
    else:
        return pred_pose_lst


def eval_metric_shapenet(cls_ids, pred_pose_lst, RTs, mask, label, obj_id, cls_name):
    #n_cls = config.n_classes
    #n_cls = 55
    #cls_add_dis = [list() for i in range(n_cls)]
    #cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    #pred_RT = torch.from_numpy(pred_RT.astype(np.float32))
    gt_RT = RTs
    
    ds_type = 'shapenet'
    
    mesh_pts = bs_utils.get_pointxyz_cuda(cls_name, obj_id, ds_type=ds_type).clone()
    #add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    #adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    add, adds = bs_utils.cal_add_and_adds_cuda(pred_RT, gt_RT, mesh_pts)
    #add, adds = bs_utils.cal_add_and_adds_cpu(pred_RT, gt_RT, mesh_pts)
    return (add.item(), adds.item())


def eval_one_frame_pose_shapenet(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id, cls_name = item
    pred_pose_lst = cal_frame_poses_shapenet(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id, cls_name
    )

    add_dis, adds_dis = eval_metric_shapenet(
        cls_ids, pred_pose_lst, RTs, mask, label, obj_id, cls_name
    )
    return (add_dis, adds_dis)

# ###############################End ShapeNet Evaluation###############################


# ###############################Shared Evaluation Entry###############################
class TorchEval():

    def __init__(self):
        self.n_cls = 2 # 1 background + 1 object
        #self.category_name = None
    
    def eval_pose_parallel(
        self, pclds, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, cls_name=None, ds='shapenet',
    ):
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        if masks is not None:
            masks = masks.long()
        else:
            masks = [None for i in range(bs)]
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        cls_name_lst = [cls_name for i in range(bs)]

        if ds == "shapenet":
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst, cls_name_lst
            )
        else:
            raise  ValueError('Currently, only shapenet dataset can be evaluated.')

        add_list=[]
        adds_list=[]

        with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as executor:
            eval_func = eval_one_frame_pose_shapenet
            for res in executor.map(eval_func, data_gen):
                add_dis, adds_dis = res
                add_list.append(add_dis)
                adds_list.append(adds_dis)
        return (add_list, adds_list)

# vim: ts=4 sw=4 sts=4 expandtab
