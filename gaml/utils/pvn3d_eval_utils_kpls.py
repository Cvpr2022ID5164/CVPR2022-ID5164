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


config = Config(ds_name='shapenet')
try:
    config_sn = Config(ds_name="shapenet")
    bs_utils_sn = Basic_Utils(config_sn)
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

# ###############################LineMOD Evaluation###############################

def cal_frame_poses_lm(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter, obj_id,
    debug=False
):
    """
    Calculates pose parameters by 3D keypoints & center points voting to build
    the 3D-3D corresponding then use least-squares fitting to get the pose parameters.
    """
    n_kps, n_pts, _ = pred_kp_of.size() # n_pts = M

    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius = 0.08
    if use_ctr:
        #cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
        cls_kps = torch.zeros(n_cls, n_kps+1, 3)
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id = 1
    #cls_msk = mask == cls_id
    kp_2ds = None

    #if cls_msk.sum() < 1:
    if n_pts < 1:
        pred_pose_lst.append(np.identity(4)[:3, :])
    else:
        cls_voted_kps = pred_kp
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr)
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
        ds_type = 'linemod'
        if obj_id > 20:
            ds_type = 'hb'

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

        kp_2ds = bs_utils.project_p3d(
            cls_kps[cls_id].cpu().numpy(), 1000.0, K=ds_type
        )

        mesh_kps = bs_utils_sn.get_kps(obj_id, ds_type=ds_type)
        if use_ctr:
            mesh_ctr = bs_utils_sn.get_ctr(obj_id, ds_type=ds_type).reshape(1, 3)
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


def eval_metric_lm(cls_ids, pred_pose_lst, RTs, mask, label, obj_id):
    #n_cls = config.n_classes
    n_cls = 55 
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    #pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32))
    gt_RT = RTs[0]
    
    ds_type = 'linemod'
    if obj_id > 20:
        ds_type = 'hb'
    
    mesh_pts = bs_utils_sn.get_pointxyz_cuda(obj_id, ds_type=ds_type).clone()
    add = bs_utils_sn.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    adds = bs_utils_sn.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    # print("obj_id:", obj_id, add, adds)
    cls_add_dis[obj_id].append(add.item())
    cls_adds_dis[obj_id].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)


def eval_one_frame_pose_lm(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst = cal_frame_poses_lm(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    cls_add_dis, cls_adds_dis = eval_metric_lm(
        cls_ids, pred_pose_lst, RTs, mask, label, obj_id
    )
    return (cls_add_dis, cls_adds_dis)

# ###############################End LineMOD Evaluation###############################


# ###############################Shared Evaluation Entry###############################
class TorchEval():

    def __init__(self):
        n_cls = 55 #22
        self.n_cls = 55
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.pred_kp_errs = [list() for i in range(n_cls)]
        self.pred_id2pose_lst = []
        self.sym_cls_ids = []

    def cal_auc(self):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        for cls_id in range(1, self.n_cls):
            if (cls_id) in config.ycb_sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = bs_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = bs_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            if i == 0:
                continue
            print(cls_lst[i-1])
            print("***************add:\t", add_auc)
            print("***************adds:\t", adds_auc)
            print("***************add(-s):\t", add_s_auc)
        # kp errs:
        n_objs = sum([len(l) for l in self.pred_kp_errs])
        all_errs = 0.0
        for cls_id in range(1, self.n_cls):
            all_errs += sum(self.pred_kp_errs[cls_id])
        print("mean kps errs:", all_errs / n_objs)

        print("Average of all object:")
        print("***************add:\t", np.mean(add_auc_lst[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst[1:]))

        print("All object (following PoseCNN):")
        print("***************add:\t", add_auc_lst[0])
        print("***************adds:\t", adds_auc_lst[0])
        print("***************add(-s):\t", add_s_auc_lst[0])

        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            pred_kp_errs=self.pred_kp_errs,
        )
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_id2pose.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(self.pred_id2pose_lst, open(sv_pth, 'wb'))

    def cal_lm_add(self, obj_id, test_occ=False):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        cls_id = obj_id
        if (obj_id) in config_lm.lm_sym_cls_ids:
            self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
        else:
            self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
        self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        add_auc = bs_utils_sn.cal_auc(self.cls_add_dis[cls_id])
        adds_auc = bs_utils_sn.cal_auc(self.cls_adds_dis[cls_id])
        add_s_auc = bs_utils_sn.cal_auc(self.cls_add_s_dis[cls_id])
        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)
        
        d_1 = config_lm.lm_r_lst[obj_id]['diameter'] / 1000.0 * 0.1
        print("obj_id: ", obj_id, "{} diameter: {} (m)".format(0.1, d_1))
        add_1 = np.mean(np.array(self.cls_add_dis[cls_id]) < d_1) * 100
        adds_1 = np.mean(np.array(self.cls_adds_dis[cls_id]) < d_1) * 100

        d_percent = 0.3
        d = config_lm.lm_r_lst[obj_id]['diameter'] / 1000.0 * d_percent
        print("obj_id: ", obj_id, "{} diameter: {} (m)".format(d_percent, d))
        add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
        adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100

        cls_type = config_lm.lm_id2obj_dict[obj_id]
        print(obj_id, cls_type)
        print("***************add auc:\t", add_auc)
        print("***************adds auc:\t", adds_auc)
        print("***************add(-s) auc:\t", add_s_auc)
        print("***************add < {} diameter:\t{}".format(0.1, add_1))
        print("***************adds < {} diameter:\t{}".format(0.1, adds_1))
        print("***************add < {} diameter:\t{}".format(d_percent, add))
        print("***************adds < {} diameter:\t{}".format(d_percent, adds))

        '''
        sv_info = dict(
            add_dis_lst=self.cls_add_dis,
            adds_dis_lst=self.cls_adds_dis,
            add_auc_lst=add_auc_lst,
            adds_auc_lst=adds_auc_lst,
            add_s_auc_lst=add_s_auc_lst,
            add=add,
            adds=adds,
        )
        occ = "occlusion" if test_occ else ""   
        sv_pth = os.path.join(
            config_lm.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_{}.pkl'.format(
                cls_type, occ, add, adds
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))
        '''
        return add
    
    def eval_pose_parallel(
        self, pclds, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, gt_kps, gt_ctrs, min_cnt=20, merge_clus=False,
        use_ctr_clus_flter=True, use_ctr=True, obj_id=0, kp_type='farthest',
        ds='ycb'
    ):
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        kp_type = [kp_type for i in range(bs)]
        if ds == "ycb":
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, gt_kps, gt_ctrs, kp_type
            )
        else:
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=bs
        ) as executor:

            eval_func = eval_one_frame_pose_lm
            for res in executor.map(eval_func, data_gen):
                cls_add_dis_lst, cls_adds_dis_lst = res

                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )

    def merge_lst(self, targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ

# vim: ts=4 sw=4 sts=4 expandtab
