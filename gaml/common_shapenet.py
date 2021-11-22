#!/usr/bin/env python3
import os
import yaml
import numpy as np
import json
import shutil

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

def filename_fix_existing(dirname, filename, moveto_pth):
    """Expands name portion of filename with numeric ' (x)' suffix to
    return filename that doesn't exist already.
    """
    fd = os.path.join(dirname, filename)
    if os.path.exists(fd) and len(os.listdir(fd)) > 0:
        all_file = os.listdir(moveto_pth)
        cur_num = len(all_file)
        moveto_file_name = os.path.join(moveto_pth, filename+'_%02d'%cur_num)
        shutil.move(fd,moveto_file_name)
    else:
        ensure_fd(fd)


class ConfigRandLA:
    k_n = 8  # KNN
    num_layers = 4  # Number of layers
    #num_points = 480 * 640 // 24  # Number of input points
    num_points = 3600
    num_classes = 42  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 1  # batch_size during training
    val_batch_size = 1  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch
    in_c = 9

    sub_sampling_ratio = [1, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [32, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 1, num_points // 4, num_points // 16, num_points // 64]


class Config:
    def __init__(self, ds_name='shapenet', cls_type=''):
        self.dataset_name = ds_name
        self.exp_dir = os.path.dirname(__file__)
        self.exp_name = os.path.basename(self.exp_dir)
        self.resnet_ptr_mdl_p = os.path.abspath(
            os.path.join(
                self.exp_dir,
                'models/cnn/ResNet_pretrained_mdl'
            )
        )
        ensure_fd(self.resnet_ptr_mdl_p)

        # log folder
        if cls_type == '':
            self.cls_type = 'all'
        else:
            self.cls_type = cls_type

        self.log_dir = os.path.abspath(
            os.path.join(self.exp_dir, 'train_log', self.dataset_name)
        )
        ensure_fd(self.log_dir)
        self.old_torch_eval_dir = os.path.abspath(
            os.path.join(self.exp_dir, 'train_log', 'old_'+self.dataset_name+'_torch_eval')
        )
        ensure_fd(self.old_torch_eval_dir)

        self.torch_eval_dir = os.path.abspath(
            os.path.join(self.exp_dir, 'train_log', self.dataset_name+'_torch_eval')
        )
        #filename_fix_existing(os.path.abspath(os.path.join(self.exp_dir, 'train_log')),
        #                      self.dataset_name+'_torch_eval',
        #                      self.old_torch_eval_dir)
        ensure_fd(self.torch_eval_dir)
        
        self.log_model_dir = os.path.join(self.log_dir, 'checkpoints', self.cls_type)
        ensure_fd(self.log_model_dir)
        self.log_eval_dir = os.path.join(self.log_dir, 'eval_results', self.cls_type)
        ensure_fd(self.log_eval_dir)
        self.log_traininfo_dir = os.path.join(self.log_dir, 'train_info', self.cls_type)
        ensure_fd(self.log_traininfo_dir)

        ##   Main Training Settings  ##
        self.use_NP = True
        self.np_type = 'local' 
        self.use_gnn_decoder = True #True # True
        if self.np_type != 'local':
            self.use_gnn_decoder = False
            # can be made more general later
            print('--------------> GNN decoder can only be used for NP_local now! <--------------')
        self.use_attention = False
        self.graph_knn = 9 # self-loop is True
        self.graph_knn_loop = True
        self.loss_type = 'L1' # 'L2', 'L1'
        self.training_context_range = [1,1] # set the range of context images for training
        #self.test_context = self.training_context_range[-1] # set #context for testing
        self.test_context = 1  # set #context for testing
        self.test_seed_per_image = 3600 # seed points per image
        self.target_seed_point = 3600 #800
        self.eval_iter = 1 #1,3,5, repeat evaluation, can be used for training and testing
        self.numpy_seed = 7 #7,17,27,37,107
        
        # Learning rate scheduler
        self.lr_dict = dict(
            lr_type='cyclic', #'constant', 'step',  'cyclic' 
            start_lr=1*1e-5,
            max_lr=5*1e-4, # for CyclicLr 
            step_percentage=0.9, # for StepLr
            gamma=0.5, # for StepLr, decay factor
        )
        self.n_total_epoch = 800
        self.eval_freq_parm = 8 # eval_epoch_freq = n_total_epoch // 2 // eval_freq_parm
        self.object_scale = 0.4
        self.n_task = 1 #3   # 1 for testing, 3 for training
        # batch size should be larger than task size!
        self.train_mini_batch_size = 2 #12 # max. 36 in the cluster
        self.val_mini_batch_size = 2 #12
        self.test_mini_batch_size = 2 #12
        self.num_instances_per_item = 48
        
        # for single category training: 
        # max_val_obj is for new test objects, max_test_obj is the training set
        self.max_train_obj = 30 # num of objects
        self.max_val_obj = 30 # num of objects
        self.max_test_obj = 30  # num of objects
        
        # For single class
        self.max_test_class = 20  # num of objects
        self.test_from_back = True 
        self.split_vis = False  # for visualization
            
        self.max_train_class = 10  # num of objects, no used for multiple classes training
        self.val_fraction = 2  # number of images, no used for multiple classes training
        assert self.val_mini_batch_size <= self.num_instances_per_item,  \
            'Check val_mini_batch_size and num_instances_per_item.'
        assert self.train_mini_batch_size <= self.num_instances_per_item, \
            'Check train_mini_batch_size and num_instances_per_item.'
        assert self.test_mini_batch_size <= self.num_instances_per_item, \
            'Check train_mini_batch_size and num_instances_per_item.'
        
        #self.n_sample_points = 480 * 640 // 24  # Number of input points
        self.n_sample_points = 3600
        self.n_keypoints = 8
        self.n_min_points = 400
        
        self.preprocessed_testset_pth = ''

        self.mcms_type = 'pbr' # 'toy', 'pbr', 'occ'
        if self.dataset_name == 'shapenet':
            self.n_objects = 1 + 1  # 1 object + background
            self.n_classes = self.n_objects

            self.shapenet_root = os.path.abspath(
                os.path.join(self.exp_dir, 'datasets/shapenet/MCMS')
            )
            self.model_info_root = os.path.join(self.shapenet_root, 'Model_info')
            self.model_xyz_root = os.path.join(self.shapenet_root, 'CAD_model')
            self.shapenet_npz_root = os.path.join(self.shapenet_root, 'Dataset',self.mcms_type)

            self.val_nid_ptn = "/data/6D_Pose_Data/datasets/LINEMOD/pose_nori_lists/{}_real_val.nori.list"

        self.intrinsic_matrix = {
            'shapenet': np.array([[375.0, 0.0, 120.0],
                                  [0.0, 375.0, 120.0],
                                  [0.0, 0.0, 1.0]])
        }

    def read_lines(self, p):
        with open(p, 'r') as f:
            return [
                line.strip() for line in f.readlines()
            ]
    def set_seed_per_image(self, num):
        self.test_seed_per_image = num

    def set_context_image(self, num):
        self.test_context = num
        self.test_mini_batch_size = self.test_context + 1

    def set_numpy_seed(self, num):
        self.numpy_seed = num

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type
        
def main():
    config = Config(ds_name='shapenet', cls_type='')


if __name__ == "__main__":
    main()
    
config = Config()
# vim: ts=4 sw=4 sts=4 expandtab