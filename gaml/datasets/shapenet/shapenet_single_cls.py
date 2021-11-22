#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
from common_shapenet import Config, ConfigRandLA
from utils.basic_utils import Basic_Utils
from ShuffleDistributedSampler import ShuffleDistributedSampler
import yaml
from glob import glob
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey
import copy

config = Config(ds_name='shapenet')
K = config.intrinsic_matrix['shapenet']

class ShapeNetData6D(object):

    def __init__(self, num_instances_per_item, ds_type='train', categ=None, drop_last=True, DEBUG=False):
        '''
        Args:
            ds_type: 'test' or 'train'
        '''
        self.image_height = 240
        self.image_width = 240
        np.random.seed(config.numpy_seed)
        self.ds_type = ds_type
        self.drop_last = drop_last
        self.DEBUG = DEBUG

        self.bs_utils = Basic_Utils(config)
        self.xmap = np.array([[j for i in range(self.image_width)] for j in range(self.image_height)])
        self.ymap = np.array([[i for i in range(self.image_width)] for j in range(self.image_height)])

        # Only consider single class
        self.cls_name = categ
        self.num_instances_per_item = num_instances_per_item  # 50
        self.val_size = config.val_fraction  #14
        self.train_size = num_instances_per_item - self.val_size # 36
        self.test_size = num_instances_per_item # 50

        self.test_img_indices = np.arange(self.test_size)
        self.train_img_indices = np.arange(self.train_size)
        self.val_img_indices = np.arange(self.val_size)

        np.random.shuffle(self.train_img_indices) # will be shuffled at each epoch

        self.test_img_indices_copy = self.test_img_indices.copy()
        self.val_img_indices_copy = self.val_img_indices.copy()
        #self.train_img_indices_copy = self.train_img_indices.copy()        
        '''
        remove_list = ['10db820f0e20396a492c7ca609cb0182',
                       '10eeb119fd5508e0d6d949577c389a84', '124a579e0635b8eace19d55bc5e6a406', 
                       '1280f994ba1f92d28699933784576e73', '12c66a0490b223be595dc3191c718398',
                       '137acaae47b50659348e240586a3f6f8', '13ea0a2ac279dbaa5e9e2656aff7dd5b',
                       '14bf5197d60d733f2a3ecc4a9713cabb', '1530400ceabc1d7145ec485f7da1d9e3',
                       '589760ac1324fcce5534ad7a7444cbeb', '5fbad7dea0243acd464e3094da7d844a', 
                       '724c6fc81a27625bb158ade66cebf744', '1039c49d2976eb87d5faf4905977884']
        '''
        self.cls_npz = np.load(os.path.join(
            config.shapenet_npz_root, '{}.npz'.format(self.cls_name)), allow_pickle=True)
        self.full_obj_list = list(self.cls_npz.files)
        
        #for obj_id in self.full_obj_list:
        #    if obj_id in remove_list:
        #        self.full_obj_list.remove(obj_id)

        print('Full dataset size = ', len(self.full_obj_list))
        self.full_test_obj_list = np.asarray(self.full_obj_list[-1 * config.max_test_class: ])
        self.test_distributed_sampler = ShuffleDistributedSampler(dataset_len=len(self.full_test_obj_list))
        indices_slice = self.test_distributed_sampler.get_indices()
        ''' 
        if ds_type=='test':
            if config.use_best_20:
                txt_f = '20_best/{}.txt'.format(self.cls_name)
                with open(txt_f) as f:
                    self.test_obj_list = [line.strip() for line in f.readlines()]
            else:
                if config.test_from_back:
                    self.test_obj_list = self.full_obj_list[-1 * config.max_test_class: ]
                    #self.test_obj_list.remove('1c1bd2dcbb13aa5a6b652ed61c4ad126')
                    self.test_obj_list = np.asarray(self.test_obj_list)
                else:
                    self.test_obj_list = np.asarray(self.full_obj_list[:config.max_test_class])
        '''
        self.test_obj_list = self.full_obj_list[-1 * config.max_test_class:]
        self.test_obj_list = np.asarray(self.test_obj_list)
        self.iter_count = {
            'train': self.train_size // config.train_mini_batch_size,
            'val': self.val_size // config.val_mini_batch_size,
            'test': self.test_size // config.test_mini_batch_size
        }
        self.estimated_it = {
            'test': self.iter_count['test'] * len(self.test_obj_list) // config.n_task
        }
        if isinstance(self.test_obj_list, list):
            self.test_counter = dict.fromkeys(self.test_obj_list, self.iter_count['test'])
        else:    
            self.test_counter = dict.fromkeys(self.test_obj_list.tolist(), self.iter_count['test'])

        if ds_type == 'train':
            # guarantee no overlapping between training and test set
            self.train_obj_list_len = min(config.max_train_class, len(self.full_obj_list) - config.max_test_class)
            self.train_batch_per_epoch = int(self.train_size * self.train_obj_list_len /
                                             (config.n_task * config.train_mini_batch_size))

            self.full_train_obj_list = np.asarray(self.full_obj_list[:self.train_obj_list_len])
            self.train_distributed_sampler = ShuffleDistributedSampler(dataset_len=self.train_obj_list_len)
            indices_slice = self.train_distributed_sampler.get_indices()
            #print('indices_slice: ', indices_slice)
            self.train_obj_list = self.full_train_obj_list[indices_slice]
            self.val_obj_list = self.full_train_obj_list[indices_slice] # won't change

            temp_train_list = self.train_obj_list.tolist() * self.iter_count['train'] # * 36//18
            self.val_iter_list = self.val_obj_list.tolist() * self.iter_count['val']
            self.test_iter_list = self.test_obj_list.tolist() * self.iter_count['test']
            np.random.shuffle(temp_train_list)
            np.random.shuffle(self.val_iter_list) # won't change
            np.random.shuffle(self.test_iter_list)  # won't change
            self.val_counter = dict.fromkeys(self.val_obj_list.tolist(), self.iter_count['val'])
            self.train_counter = dict.fromkeys(self.train_obj_list.tolist(), self.iter_count['train'])
            self.estimated_it['val'] = self.iter_count['val'] * len(self.val_obj_list) // config.n_task
            self.estimated_it['train'] = self.iter_count['train'] * len(self.train_obj_list) // config.n_task

            if drop_last:
                self.train_iter = iter(self.__drop_last_truncation(temp_train_list,config.n_task)) # -> Dataloader
                self.val_iter = iter(self.__drop_last_truncation(self.val_iter_list, config.n_task))  # -> Dataloader
                self.test_iter = iter(self.__drop_last_truncation(self.test_iter_list, config.n_task))  # -> Dataloader
            else:
                self.train_iter = iter(temp_train_list)
                self.val_iter = iter(self.val_iter_list)
                self.test_iter = iter(self.test_iter_list)

            print('-----> {}_dataset: train {} objects, estimated iteration = {}'.format(
                ds_type, self.train_obj_list_len, self.train_batch_per_epoch))
            print(self.estimated_it)
        elif ds_type == 'test':
            #print('repeat ',self.iter_count['test'] )
            self.test_iter_list = np.repeat(self.test_obj_list, self.iter_count['test']) # test object by object
            self.test_iter_list = self.test_iter_list.tolist()
            #print(self.test_iter_list)
            self.test_iter = iter(self.test_iter_list) # n_task = 1 during testing
        else:
            raise ValueError('Given dataset type is not defined. Choose between \'train\' and \'test\'')

    def __drop_last_truncation(self, a_list, portion_per_block):
        list_len = len(a_list)
        num_block = list_len // portion_per_block
        return a_list[:num_block * portion_per_block]

    def get_batch(self, source, tasks_per_batch):
        """
        Wrapper function for batching in the model.
        :param source: train, validation or test (string).
        :param tasks_per_batch: number of tasks to include in batch.
        :return: dict
        """
        batch_dict_list = []
        if source == 'train':
            temp_iter = self.train_iter
            mini_batch_size = config.train_mini_batch_size
            all_batch_indices = self.train_img_indices
            batch_counter = self.train_counter
        elif source == 'val':
            temp_iter = self.val_iter
            mini_batch_size = config.val_mini_batch_size
            all_batch_indices = self.val_img_indices_copy
            batch_counter = self.val_counter
        elif source == 'test':
            temp_iter = self.test_iter
            mini_batch_size = config.test_mini_batch_size
            all_batch_indices = self.test_img_indices_copy
            batch_counter = self.test_counter
        else:
            raise ValueError('Given dataset mode is not defined.')

        max_count = self.iter_count[source]
        all_idx = [*range(mini_batch_size * tasks_per_batch)]
        ordered_idx = [0] * (mini_batch_size * tasks_per_batch)

        for i in range(tasks_per_batch):
            try:
                current_obj = next(temp_iter)
            except StopIteration:
                return None
            k  = max_count - batch_counter[current_obj]
            batch_indices = all_batch_indices[k*mini_batch_size: (k+1)*mini_batch_size]
            np.random.shuffle(batch_indices)
            batch_counter[current_obj] = batch_counter[current_obj] - 1

            task_dict = self.get_item_batch_wise(current_obj,batch_indices)
            temp_idx = all_idx[i * mini_batch_size: i * mini_batch_size + mini_batch_size]
            ordered_idx[i::tasks_per_batch] = temp_idx
            batch_dict_list.append(task_dict)
        #print('ordered_idx = ', ordered_idx)
        batch_dict = self.__dict_list2dict(batch_dict_list, expand_dim=False, order=ordered_idx)
        #print(batch_dict['cls_id'])
        return batch_dict

    def get_item_batch_wise(self, obj_name, batch_indices):
        '''
        Args:
            mode: 'train','test','val'
        Returns: dictionary
        '''
        obj_arr = self.cls_npz[obj_name]
        obj_instances = obj_arr[batch_indices]

        item_dict_list = []
        for i in range(len(batch_indices)):
            item_dict_list.append(self.get_item(obj_name, obj_instances[i]))

        batch_dict = self.__dict_list2dict(item_dict_list, expand_dim=True)
        #print(key, batch_dict[key].shape)
        return batch_dict

    def get_item(self, obj_name, item_arr):
        rgb = item_arr['rgb']
        labels = item_arr['mask'][:,:,0]
        #item_index = obj_instances['item_index']
        rgb_labels= labels.copy()
        RT = item_arr['RT']
        cam_scale = 1000
        dpt_mm = item_arr['depth_mm']
        cls_name = item_arr['cls_name']
        dpt_m = dpt_mm / 1000
        msk_dp = dpt_mm > 1e-6

        #nrm_map = normalSpeed.depth_normal(
        #    dpt_mm, K[0][0], K[1][1], 5, 15000, 500, False
        #)
        nrm_map = item_arr['normal']

        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0
        if self.DEBUG:
            real_mask = labels.flatten().nonzero()[0].astype(np.uint32)

        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        if len(choose) > config.n_sample_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose[c_mask.nonzero()]
        else:
            choose_2 = np.array([i for i in range(len(choose))])
            choose_2 = np.pad(choose_2, (0, config.n_sample_points - len(choose_2)), 'wrap')

        choose = np.array(choose)[choose_2]
        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)
        choose = np.array([choose])
        if self.DEBUG:
            show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            # imshow("nrm_map", show_nrm_map)
            original_rgb = rgb[..., ::-1].copy()
            imshow('minitest', original_rgb)
            p2ds = self.bs_utils.project_p3d(cld, cam_scale, K)
            show_rgb = self.bs_utils.paste_p2ds(original_rgb.copy(), p2ds, (0, 0, 255))
            print(p2ds.shape)
            imshow("test point clod image", show_rgb)
            cmd = waitKey(0)
            if cmd == ord('q'):
                exit()

        kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst, kps_ctr = self.get_pose_gt_info(
            cld, labels_pt, RT, obj_name, self.cls_name)

        h, w= rgb_labels.shape
        #dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w
        #msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            #msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0)
            for ii, item in enumerate(xyz_lst)
        }

        rgb_ds_sr = [4, 8, 8 ,8] # TODO: should be the same as network
        n_ds_layers = 4
        pcld_sub_s_r = [1, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 8
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2] #TODO： need to check
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

        show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]
        if self.DEBUG:
            for ip, xyz in enumerate(xyz_lst):
                pcld = xyz.reshape(3, -1).transpose(1, 0)
                p2ds = self.bs_utils.project_p3d(pcld, cam_scale, K)
                srgb = self.bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                #imshow("rz_pcld_%d" % ip, srgb)
                p2ds = self.bs_utils.project_p3d(inputs['cld_xyz%d' % ip], cam_scale, K)
                print(ip, p2ds.shape)
                self.check_pcld_on_obj(real_mask, p2d=p2ds)
                srgb1 = self.bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                imshow("rz_pcld_%d_rnd" % ip, srgb1)
        # print(
        #     "kp3ds:", kp3ds.shape, kp3ds, "\n",
        #     "kp3ds.mean:", np.mean(kp3ds, axis=0), "\n",
        #     "ctr3ds:", ctr3ds.shape, ctr3ds, "\n",
        #     "cls_ids:", cls_ids, "\n",
        #     "labels.unique:", np.unique(labels),
        # )
        item_dict = dict(
            cls_name=np.array([self.cls_name]),
            obj_name=np.array([obj_name]),
            cls_id=np.array(item_arr['item_index']).astype(np.uint16),
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
            RTs=RT.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
            kps_ctr=kps_ctr.astype(np.float32),
        )
        item_dict.update(inputs)
        return item_dict

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def get_pose_gt_info(self, cld, labels, RT, obj_id, cls_name):
        RTs = np.zeros((config.n_objects, 3, 4))
        kp3ds = np.zeros((config.n_objects, config.n_keypoints, 3))
        ctr3ds = np.zeros((config.n_objects, 3))
        cls_ids = np.zeros((config.n_objects, 1))
        kp_targ_ofst = np.zeros((config.n_sample_points, config.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((config.n_sample_points, 3))
        kps_ctr = []

        for i, cls_id in enumerate([1]):
            RTs[i] = RT
            r = RT[:, :3]
            t = RT[:, 3]

            ctr = self.bs_utils.get_ctr(cls_name, obj_id, ds_type="shapenet")[:, None]
            kps_ctr.append(ctr.T)
            ctr = np.dot(ctr.T, r.T) + t
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([1])

            kps = self.bs_utils.get_kps(cls_name, obj_id, ds_type='shapenet')
            kps_ctr.insert(0, kps)
            kps = np.dot(kps, r.T) + t
            kp3ds[i] = kps

            kps_ctr = np.concatenate(kps_ctr)

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0*kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

        return kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst, kps_ctr

    def check_pcld_on_obj(self, real_mask, p2d=None):
        count = 0
        for i in range(p2d.shape[0]):
            index = p2d[i,0] * self.image_width + p2d[i,1]
            if index in real_mask:
                count += 1
        print('{} points on the object!'.format(count))

    def __dict_list2dict(self, dict_list, expand_dim=False, order=None):
        gather_dict = {}
        for key in dict_list[0].keys():
            cat_list = []
            for item in dict_list:
                if expand_dim:
                    cat_list.append(item[key][None, ...])
                else:
                    cat_list.append(item[key])
            gather_dict[key] = np.concatenate(cat_list, axis=0)
            if order is not None:
                gather_dict[key] = gather_dict[key][order]
        return gather_dict

    def __random_indices_generator(self, target, epoch=0, mode=None):
        if mode == 'test':
            k = 2
        elif mode == 'val':
            k = 5
        else:
            k = 10
        g = torch.Generator()
        g.manual_seed(config.numpy_seed + k * epoch)
        indices = torch.randperm(len(target), generator=g).tolist()  # type: ignore
        return indices

    def reset_set_epoch(self, source, epoch=0, shuffle_obj=False):
        if source == 'train':
            self.__set_train_epoch(epoch, shuffle_obj)
        elif source == 'val':
            self.__reset_val_iter(epoch, shuffle_obj)
        elif source == 'test':
            self.__reset_test_iter(epoch, shuffle_obj)
        else:
            raise ValueError('Given dataset mode is not defined.')

    def __set_train_epoch(self, epoch, shuffle_obj=True):
        np.random.shuffle(self.train_img_indices) # shuffle images within the same object list
        self.train_distributed_sampler.set_epoch(epoch)
        indices_slice = self.train_distributed_sampler.get_indices() # shuffle object list and re-distributed across GPUs
        #print('indices_slice: ', indices_slice)
        self.train_obj_list = self.full_train_obj_list[indices_slice]
        self.train_counter = dict.fromkeys(self.train_obj_list.tolist(), self.iter_count['train'])
        temp_train_list = self.train_obj_list.tolist() * (self.train_size // config.train_mini_batch_size)  # * 36//18
        np.random.shuffle(temp_train_list)
        if self.drop_last:
            self.train_iter = iter(self.__drop_last_truncation(temp_train_list, config.n_task))  # -> Dataloader
        else:
            self.train_iter = iter(temp_train_list)

    def __reset_val_iter(self, epoch=0, shuffle_obj=False):
        indices = self.__random_indices_generator(self.val_img_indices, epoch,'val')
        self.val_img_indices_copy = self.val_img_indices[indices]
        self.val_counter = dict.fromkeys(self.val_obj_list.tolist(), self.iter_count['val'])
        if self.drop_last:
            self.val_iter = iter(self.__drop_last_truncation(self.val_iter_list, config.n_task))  # -> Dataloader
        else:
            self.val_iter = iter(self.val_iter_list)  # -> Dataloader

    def __reset_test_iter(self, epoch=0, shuffle_obj=False):
        if shuffle_obj:
            self.test_distributed_sampler.set_epoch(epoch)
            indices_slice = self.test_distributed_sampler.get_indices() # shuffle object list and re-distributed across GPUs
            self.test_obj_list = self.full_test_obj_list[indices_slice]
            self.test_iter_list = self.test_obj_list.tolist() * self.iter_count['test']
        indices = self.__random_indices_generator(self.test_img_indices, epoch, 'test')
        self.test_img_indices_copy = self.test_img_indices[indices]
        
        if isinstance(self.test_obj_list, list):
            self.test_counter = dict.fromkeys(self.test_obj_list, self.iter_count['test'])
        else:    
            self.test_counter = dict.fromkeys(self.test_obj_list.tolist(), self.iter_count['test'])
        if self.drop_last:
            self.test_iter = iter(self.__drop_last_truncation(self.test_iter_list, config.n_task))  # -> Dataloader
        else:
            self.test_iter = iter(self.test_iter_list)  # -> Dataloader

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--gpu', type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )
    ds_type = 'test'
    cls_name = 'car'
    ds = ShapeNetData6D(num_instances_per_item=4, ds_type=ds_type, categ=cls_name, DEBUG=False)
    ds.reset_set_epoch('test', 2, shuffle_obj=True)
    config_temp = Config(ds_name='shapenet', cls_type=cls_name)
    bs_utils = Basic_Utils(config_temp)

    while True:
        data = ds.get_batch(source='test', tasks_per_batch=2)
        if data is None:
            break
        bs = data['obj_name'].shape[0]
        for i in range(bs):
            kp3d = data['kp_3ds'][i]
            kp3d = kp3d[0]
            kp_2ds = bs_utils.project_p3d(kp3d, 1000, K)
            np_rgb = data['rgb'][i].transpose(1, 2, 0).copy()
            ori_rgb = np_rgb.copy()
            rgb = bs_utils.draw_p2ds(ori_rgb, kp_2ds, color=(0,0,255),thickness=2)
            imshow('minitest', rgb)
            cmd = waitKey(0)
            if cmd == ord('q'):
                exit()
            else:
                continue


if __name__ == "__main__":
    main()