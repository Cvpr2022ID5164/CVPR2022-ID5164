#!/usr/bin/env python3
import os
import cv2
import random
import torch
import numpy as np

from plyfile import PlyData
import normalSpeed

from utils.ip_basic.ip_basic import vis_utils
from utils.ip_basic.ip_basic import depth_map_utils_ycb as depth_map_utils


intrinsic_matrix = {
    'shapenet': np.array([[375.0, 0.0, 120.0],
                          [0.0, 375.0, 120.0],
                          [0.0, 0.0, 1.0]])
}


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


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

class Basic_Utils():

    def __init__(self, config):
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.config = config
        self.shapenet_cls_kps_dict = {'car':{}}
        self.shapenet_cls_ctr_dict = {'car':{}}
        self.shapenet_xyz_npz_dict = {}
        self.shapenet_cls_ptsxyz_dict = {}
        self.shapenet_cls_ptsxyz_cuda_dict = {}


    def read_lines(self, p):
        with open(p, 'r') as f:
            lines = [
                line.strip() for line in f.readlines()
            ]
        return lines

    def sv_lines(self, p, line_lst):
        with open(p, 'w') as f:
            for line in line_lst:
                print(line, file=f)

    def translate(self, img, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return shifted

    def rotate(self, img, angle, ctr=None, scale=1.0):
        (h, w) = img.shape[:2]
        if ctr is None:
            ctr = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(ctr, -1.0 * angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated

    def cal_degree_from_vec(self, v1, v2):
        cos = np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if abs(cos) > 1.0:
            cos = 1.0 * (-1.0 if cos < 0 else 1.0)
            print(cos, v1, v2)
        dg = np.arccos(cos) / np.pi * 180
        return dg

    def cal_directional_degree_from_vec(self, v1, v2):
        dg12 = self.cal_degree_from_vec(v1, v2)
        cross = v1[0] * v2[1] - v2[0] * v1[1]
        if cross < 0:
            dg12 = 360 - dg12
        return dg12

    def mean_shift(self, data, radius=5.0):
        clusters = []
        for i in range(len(data)):
            cluster_centroid = data[i]
            cluster_frequency = np.zeros(len(data))
            # Search points in circle
            while True:
                temp_data = []
                for j in range(len(data)):
                    v = data[j]
                    # Handle points in the circles
                    if np.linalg.norm(v - cluster_centroid) <= radius:
                        temp_data.append(v)
                        cluster_frequency[i] += 1
                # Update centroid
                old_centroid = cluster_centroid
                new_centroid = np.average(temp_data, axis=0)
                cluster_centroid = new_centroid
                # Find the mode
                if np.array_equal(new_centroid, old_centroid):
                    break
            # Combined 'same' clusters
            has_same_cluster = False
            for cluster in clusters:
                if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
                    has_same_cluster = True
                    cluster['frequency'] = cluster['frequency'] + cluster_frequency
                    break
            if not has_same_cluster:
                clusters.append({
                    'centroid': cluster_centroid,
                    'frequency': cluster_frequency
                })

        print('clusters (', len(clusters), '): ', clusters)
        self.clustering(data, clusters)
        return clusters

    # Clustering data using frequency
    def clustering(self, data, clusters):
        t = []
        for cluster in clusters:
            cluster['data'] = []
            t.append(cluster['frequency'])
        t = np.array(t)
        # Clustering
        for i in range(len(data)):
            column_frequency = t[:, i]
            cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
            clusters[cluster_index]['data'].append(data[i])

    def project_p3d(self, p3d, cam_scale, K=intrinsic_matrix['shapenet']):
        if type(K) == str:
            K = intrinsic_matrix[K]
        p3d = p3d * cam_scale
        p2d = np.dot(p3d, K.T)
        p2d_3 = p2d[:, 2]
        p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
        p2d[:, 2] = p2d_3
        p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
        return p2d

    def ensure_dir(self, pth):
        if not os.path.exists(pth):
            os.system("mkdir -p %s" % pth)

    def draw_p2ds(self, img, p2ds, r=1, thickness=1, color=[(255, 0, 0)], 
                  filled=False):
        if type(color) == tuple:
            color = [color]
        if len(color) < p2ds.shape[0]:
            color = [color[0] for i in range(p2ds.shape[0])]
        elif len(color)>p2ds.shape[0]:
            color = [color[i] for i in range(p2ds.shape[0])]

        h, w = img.shape[0], img.shape[1]
        for pt_2d, c in zip(p2ds, color):
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            if filled:   
                img = cv2.circle(
                    img, (pt_2d[0], pt_2d[1]), r, c, thickness, cv2.FILLED
                )
            else:
                img = cv2.circle(
                    img, (pt_2d[0], pt_2d[1]), r, c, thickness
                )                
        return img

    def draw_p2ds_alpha(self, img, p2ds, r=1, thickness=1, color=[(255, 0, 0)], 
                  filled=False, alpha=0.5):
        if type(color) == tuple:
            color = [color]
        if len(color) < p2ds.shape[0]:
            color = [color[0] for i in range(p2ds.shape[0])]
        elif len(color)>p2ds.shape[0]:
            color = [color[i] for i in range(p2ds.shape[0])]

        h, w = img.shape[0], img.shape[1]
        
        out = img.copy()
        # Initialize blank mask image of same dimensions for drawing the shapes
        shapes = img.copy() #np.zeros_like(img, np.uint8)
        #print('1111')
        for pt_2d, c in zip(p2ds, color):
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            if filled:   
                shapes = cv2.circle(
                    shapes, (pt_2d[0], pt_2d[1]), r, c, -1
                )
                #print('222221')
                if alpha is not None:
                    # Generate output by blending image with shapes image, using the shapes
                    # images also as mask to limit the blending to those parts
                    mask = img.astype(bool)
                    out[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]
            else:
                shapes = cv2.circle(
                    shapes, (pt_2d[0], pt_2d[1]), r, c, thickness
                )                
        return out

    def draw_p2ds_rect(self, img, p2ds, r=1, color=[(255, 0, 0)]):
        if type(color) == tuple:
            color = [color]
        if len(color) < p2ds.shape[0]:
            color = [color[0] for i in range(p2ds.shape[0])]
        elif len(color)>p2ds.shape[0]:
            color = [color[i] for i in range(p2ds.shape[0])]

        h, w = img.shape[0], img.shape[1]
        for pt_2d, c in zip(p2ds, color):
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            img = cv2.rectangle(
                img, (pt_2d[0]-r, pt_2d[1]+r), (pt_2d[0]+r, pt_2d[1]-r), c, -1
            )
        return img


    def draw_p2ds_triangle(self, img, p2ds, r=1, color=[(255, 0, 0)]):
        if type(color) == tuple:
            color = [color]
        if len(color) < p2ds.shape[0]:
            color = [color[0] for i in range(p2ds.shape[0])]
        elif len(color)>p2ds.shape[0]:
            color = [color[i] for i in range(p2ds.shape[0])]

        h, w = img.shape[0], img.shape[1]
        for pt_2d, c in zip(p2ds, color):
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            vertices= np.array([[pt_2d[0], pt_2d[1]+r],
                [int(pt_2d[0]+np.sqrt(3)*r/2), int(pt_2d[1]-r/2)],
                [int(pt_2d[0]-np.sqrt(3)*r/2), int(pt_2d[1]-r/2)]],np.int32)
            pts = vertices.reshape((-1, 1, 2))
            img = cv2.fillPoly(img,[pts],c)
        return img

    def paste_p2ds(self, img, p2ds, color=[(255, 0, 0)]):
        if type(color) == tuple:
            color = [color]
        if len(color) != p2ds.shape[0]:
            color = [color[0] for i in range(p2ds.shape[0])]
        h, w = img.shape[0], img.shape[1]
        p2ds[:, 0] = np.clip(p2ds[:, 0], 0, w)
        p2ds[:, 1] = np.clip(p2ds[:, 1], 0, h)
        img[p2ds[:, 1], p2ds[:, 0]] = np.array(color)
        return img

    def draw_p2ds_lb(self, img, p2ds, label, r=1, color=(255, 0, 0)):
        h, w = img.shape[0], img.shape[1]
        for pt_2d, lb in zip(p2ds, label):
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            color = self.get_label_color(lb)
            img = cv2.circle(
                img, (pt_2d[0], pt_2d[1]), r, color, -1
            )
        return img

    def quick_nrm_map(
        self, dpt, scale_to_mm, K=intrinsic_matrix['shapenet'], with_show=False
    ):
        dpt_mm = (dpt.copy() * scale_to_mm).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)
        if with_show:
            nrm_map[np.isnan(nrm_map)] = 0.0
            nrm_map[np.isinf(nrm_map)] = 0.0
            show_nrm = ((nrm_map[:, :, :3] + 1.0) * 127).astype(np.uint8)
            return nrm_map, show_nrm
        return nrm_map

    def dpt_2_showdpt(self, dpt, scale2m=1.0):
        min_d, max_d = dpt[dpt > 0].min(), dpt.max()
        dpt[dpt > 0] = (dpt[dpt > 0]-min_d) / (max_d - min_d) * 255
        # dpt = (dpt / scale2m) / dpt.max() * 255 #127
        dpt = dpt.astype(np.uint8)
        im_color = cv2.applyColorMap(
            cv2.convertScaleAbs(dpt, alpha=1), cv2.COLORMAP_JET
        )
        return im_color

    def get_show_label_img(self, labels, mode=1):
        cls_ids = np.unique(labels)
        n_obj = np.max(cls_ids)
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]
        h, w = labels.shape
        show_labels = np.zeros(
            (h, w, 3), dtype='uint8'
        )
        labels = labels.reshape(-1)
        show_labels = show_labels.reshape(-1, 3)
        for cls_id in cls_ids:
            if cls_id == 0:
                continue
            cls_color = np.array(
                self.get_label_color(cls_id, n_obj=n_obj, mode=mode)
            )
            show_labels[labels == cls_id, :] = cls_color
        show_labels = show_labels.reshape(h, w, 3)
        return show_labels

    def get_label_color(self, cls_id, n_obj=22, mode=0):
        if mode == 0:
            cls_color = [
                255, 255, 255,  # 0
                180, 105, 255,   # 194, 194, 0,    # 1 # 194, 194, 0
                0, 255, 0,      # 2
                0, 0, 255,      # 3
                0, 255, 255,    # 4
                255, 0, 255,    # 5
                180, 105, 255,  # 128, 128, 0,    # 6
                128, 0, 0,      # 7
                0, 128, 0,      # 8
                0, 165, 255,    # 0, 0, 128,      # 9
                128, 128, 0,    # 10
                0, 0, 255,      # 11
                255, 0, 0,      # 12
                0, 194, 0,      # 13
                0, 194, 0,      # 14
                255, 255, 0,    # 15 # 0, 194, 194
                64, 64, 0,      # 16
                64, 0, 64,      # 17
                185, 218, 255,  # 0, 0, 64,       # 18
                0, 0, 255,      # 19
                0, 64, 0,       # 20
                0, 0, 192       # 21
            ]
            cls_color = np.array(cls_color).reshape(-1, 3)
            color = cls_color[cls_id]
            bgr = (int(color[0]), int(color[1]), int(color[2]))
        elif mode == 1:
            cls_color = [
                255, 255, 255,  # 0
                0, 127, 255,    # 180, 105, 255,   # 194, 194, 0,    # 1 # 194, 194, 0
                0, 255, 0,      # 2
                255, 0, 0,      # 3
                180, 105, 255, # 0, 255, 255,    # 4
                255, 0, 255,    # 5
                180, 105, 255,  # 128, 128, 0,    # 6
                128, 0, 0,      # 7
                0, 128, 0,      # 8
                185, 218, 255,# 0, 0, 255, # 0, 165, 255,    # 0, 0, 128,      # 9
                128, 128, 0,    # 10
                0, 0, 255,      # 11
                255, 0, 0,      # 12
                0, 194, 0,      # 13
                0, 194, 0,      # 14
                255, 255, 0,    # 15 # 0, 194, 194
                0, 0, 255, # 64, 64, 0,      # 16
                64, 0, 64,      # 17
                185, 218, 255,  # 0, 0, 64,       # 18
                0, 0, 255,      # 19
                0, 0, 255, # 0, 64, 0,       # 20
                0, 255, 255,# 0, 0, 192       # 21
            ]
            cls_color = np.array(cls_color).reshape(-1, 3)
            color = cls_color[cls_id]
            bgr = (int(color[0]), int(color[1]), int(color[2]))
        elif mode == 2:
            cls_color = [
                228, 26, 28,  # red
                55, 126, 184, # blue
                255, 127, 0,  # orange
                77, 175, 74, # green
                152, 78, 163, # purple
                255, 255, 51, # yellow
                166, 86, 40 # brawn
            ]
            cls_color = np.array(cls_color).reshape(-1, 3)
            color = cls_color[cls_id]
            bgr = (int(color[0]), int(color[1]), int(color[2]))
        else:
            mul_col = 255 * 255 * 255 // n_obj * cls_id
            r, g, b= mul_col // 255 // 255, (mul_col // 255) % 255, mul_col % 255
            bgr = (int(r), int(g) , int(b))
        return bgr

    def dpt_2_cld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        msk_dp = dpt > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 1:
            return None, None

        dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_mskd = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_mskd = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        pt2 = dpt_mskd / cam_scale
        cam_cx, cam_cy = K[0][2], K[1][2]
        cam_fx, cam_fy = K[0][0], K[1][1]
        pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
        cld = np.concatenate((pt0, pt1, pt2), axis=1)
        return cld, choose

    def get_normal_map(self, nrm, choose):
        nrm_map = np.zeros((480, 640, 3), dtype=np.uint8)
        nrm = nrm[:, :3]
        nrm[np.isnan(nrm)] = 0.0
        nrm[np.isinf(nrm)] = 0.0
        nrm_color = ((nrm + 1.0) * 127).astype(np.uint8)
        nrm_map = nrm_map.reshape(-1, 3)
        nrm_map[choose, :] = nrm_color
        nrm_map = nrm_map.reshape((480, 640, 3))
        return nrm_map

    def get_rgb_pts_map(self, pts, choose):
        pts_map = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = pts[:, :3]
        pts[np.isnan(pts)] = 0.0
        pts[np.isinf(pts)] = 0.0
        pts_color = pts.astype(np.uint8)
        pts_map = pts_map.reshape(-1, 3)
        pts_map[choose, :] = pts_color
        pts_map = pts_map.reshape((480, 640, 3))
        return pts_map

    def fill_missing(
            self, dpt, cam_scale, scale_2_80m, fill_type='multiscale',
            extrapolate=False, show_process=False, blur_type='bilateral'
    ):
        dpt = dpt / cam_scale * scale_2_80m
        projected_depth = dpt.copy()
        if fill_type == 'fast':
            final_dpt = depth_map_utils.fill_in_fast(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                # max_depth=2.0
            )
        elif fill_type == 'multiscale':
            final_dpt, process_dict = depth_map_utils.fill_in_multiscale(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process,
                max_depth=3.0
            )
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        dpt = final_dpt / scale_2_80m * cam_scale
        return dpt

    def rand_range(self, lo, hi):
        return random.random()*(hi-lo)+lo

    def get_ycb_ply_mdl(
        self, cls
    ):
        ply_pattern = os.path.join(
            self.config.ycb_root, '/models',
            '{}/textured.ply'
        )
        ply = PlyData.read(ply_pattern.format(cls, cls))
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        return model

    def get_cls_name(self, cls, ds_type):
        if type(cls) is int:
            if ds_type == 'ycb':
                cls = self.ycb_cls_lst[cls - 1]
            else:
                cls = self.lm_cls_lst[cls - 1]
        return cls

    def ply_vtx(self, pth, ds_type='linemod'):
        f = open(pth)
        assert f.readline().strip() == "ply"
        f.readline()
        if ds_type=='linemod':
            f.readline()
        N = int(f.readline().split()[-1])
        while f.readline().strip() != "end_header":
            continue
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
        return np.array(pts)

    def get_pointxyz(self, cls, obj_name, ds_type='shapenet', 
                     mode='full',choose_samples=10000):
        # mode: full, sample
        if ds_type=='shapenet':
            if cls not in self.shapenet_xyz_npz_dict.keys():
                xyz_path = os.path.join(
                    self.config.model_xyz_root,'{}.npz'.format(cls))
                xyz_npz = np.load(xyz_path, allow_pickle=True)
                out_xyz = xyz_npz[obj_name]
                self.shapenet_xyz_npz_dict.update(
                    {cls: xyz_npz}
                )
            else:
                xyz_npz = self.shapenet_xyz_npz_dict[cls]
                out_xyz = xyz_npz[obj_name]
                #print('-------> out xyz shape: ', out_xyz.shape)
                if mode=='sample' and out_xyz.shape[0] > choose_samples:
                    out_xyz = out_xyz[
                        np.random.choice(out_xyz.shape[0], choose_samples, replace=False)
                    ]
                    #print('-------> Sampled: out xyz shape: ', out_xyz.shape)
            return out_xyz * self.config.object_scale
        else:
            raise ValueError('get_pointxyz: dataset name is not defined.')

    def get_pointxyz_cuda(self, cls, obj_name, ds_type='shapenet'):
            ptsxyz = self.get_pointxyz(cls, obj_name, ds_type)
            ptsxyz_cu = torch.from_numpy(ptsxyz.astype(np.float32))
            return ptsxyz_cu.clone()

    def get_kps(
        self, cls, obj_id, ds_type='shapenet', kp_pth=None
    ):
        if kp_pth:
            kps = np.loadtxt(kp_pth, dtype=np.float32)
            return kps

        if ds_type == 'shapenet':
            if cls not in self.shapenet_cls_kps_dict.keys():
                self.shapenet_cls_kps_dict.update({cls: {} })
            else:
                if obj_id in self.shapenet_cls_kps_dict[cls].keys():
                    return self.shapenet_cls_kps_dict[cls][obj_id].copy()
            #print("kps_pth in get_kps:", kps_pth)
            kps_pth = os.path.join(
                self.config.model_info_root, cls, obj_id,
                '{}_8_kps.txt'.format(obj_id)
            )
            kps = np.loadtxt(kps_pth, dtype=np.float32) * self.config.object_scale
            self.shapenet_cls_kps_dict[cls][obj_id] = kps
            return kps.copy()
        else:
            raise ValueError('get_kps: only can be used by ShapeNet.')


    def get_ctr(self, cls, obj_id, ds_type='shapenet', ctr_pth=None):
        if ctr_pth:
            ctr = np.loadtxt(ctr_pth, dtype=np.float32)
            return ctr
        if ds_type == 'shapenet':
            if cls not in self.shapenet_cls_ctr_dict.keys():
                self.shapenet_cls_ctr_dict.update({cls: {} })
            else:
                if obj_id in self.shapenet_cls_ctr_dict[cls].keys():
                    return self.shapenet_cls_ctr_dict[cls][obj_id].copy()
            ctr_path = os.path.join(
                self.config.model_info_root, cls, obj_id,
                '{}_corners.txt'.format(obj_id)
            )
            cors = np.loadtxt(ctr_path, dtype=np.float32)
            ctr = cors.mean(0) * self.config.object_scale
            self.shapenet_cls_ctr_dict[cls][obj_id] = ctr
        else:
            raise ValueError('get_ctr: only can be used by ShapeNet.')
        return ctr.copy()

    def get_diameter(self, cls, obj_id, ds_type='shapenet'):
        if ds_type == 'shapenet':
            r_path = os.path.join(
                self.config.model_info_root, cls, obj_id,
                '{}_radius.txt'.format(obj_id)
            )
            radius = np.loadtxt(r_path, dtype=np.float32)
        else:
            raise ValueError('get_ctr: only can be used by ShapeNet.')
        return radius * 2.0 * self.config.object_scale

    def cal_auc(self, add_dis, max_dis=0.1):
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf;
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = VOCap(D, acc)
        return aps * 100

    def cal_add_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        p3ds = p3ds.cuda() #.cpu()
        pred_RT=pred_RT.cuda() #.cpu()
        gt_RT = gt_RT.cuda() #.cpu()

        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
        return torch.mean(dis)

    def cal_adds_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        N, _ = p3ds.size()

        p3ds = p3ds.cuda() #.cpu()
        pred_RT=pred_RT.cuda() #.cpu()
        gt_RT = gt_RT.cuda() #.cpu()

        pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        pd = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        gt = gt.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        return torch.mean(mdis)

    
    def cal_add_and_adds_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        N, _ = p3ds.size()
          
        if N > 15000:
            p3ds = p3ds[torch.randint(N, (15000,))]
            N = p3ds.shape[0]
            #print('---------> new N=',N)
        p3ds = p3ds.cuda() #.cpu()
        pred_RT=pred_RT.cuda() #.cpu()
        gt_RT = gt_RT.cuda() #.cpu()

        # calculate add
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
        add =  torch.mean(dis)
        del dis
        del gt_p3ds
        del pred_p3ds
        
        # calculate adds
        pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        pd = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        gt = gt.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        adds = torch.mean(mdis)    
        del p3ds 
        del pred_RT
        del gt_RT 
        del pd
        del gt
        del mdis
        return add, adds    
    
    
    def best_fit_transform_torch(self, A, B):
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
        assert A.size() == B.size()
        # get number of dimensions
        m = A.size()[1]
        # translate points to their centroids
        centroid_A = torch.mean(A, dim=0)
        centroid_B = torch.mean(B, dim=0)
        AA = A - centroid_A
        BB = B - centroid_B
        # rotation matirx
        H = torch.mm(AA.transpose(1, 0), BB)
        U, S, Vt = torch.svd(H)
        R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
        # special reflection case
        if torch.det(R) < 0:
            Vt[m-1, :] *= -1
            R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
        # translation
        t = centroid_B - torch.mm(R, centroid_A.view(3, 1))[:, 0]
        #T = torch.zeros(3, 4).cuda()
        T = torch.zeros(3, 4)
        T[:, :3] = R
        T[:, 3] = t
        return  T

    def best_fit_transform(self, A, B):
        return best_fit_transform(A, B)


if __name__ == "__main__":

    pass
# vim: ts=4 sw=4 sts=4 expandtab
