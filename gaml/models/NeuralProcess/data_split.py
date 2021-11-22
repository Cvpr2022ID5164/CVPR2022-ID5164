import torch
import random
#random.seed(7)
from six.moves import cPickle as pickle
from common_shapenet import Config
import argparse

# for GNN decoder

from torch_cluster import knn_graph
from torch_geometric.data import Batch
from models.gnn_utils import BatchData

class ContextTargetSplit:

    def __init__(self, n_kps=8,config=None):
        super().__init__()
        
        if config is None:
            self.config = Config(ds_name='shapenet')
        else:
            self.config = config
            
        self.n_kps = n_kps
        self.context_range = self.config.training_context_range


    def context_target_split(self, rgbd_emb, gt_kp_ofst, gt_ctr_ofst,
                             labels, gt_labels, kps_ctr=None, max_num=50, min_num=30,
                             cld=None, is_training=True, c_in_t=False):
        cls_id = 1
        #cls_msk = labels == cls_id
        M_t = self.config.target_seed_point

        # find the negative mask prediction
        #wrong_predict = torch.logical_xor(labels,gt_labels)
        #wrong_predict = torch.logical_and(wrong_predict,labels) # (bs,3600) # value 1 means it's the wrong prediction

        bs, M = gt_labels.shape

        # random context images during training, fixed amount during testing
        if is_training:
            n_context_image = random.randint(self.context_range[0], self.context_range[1]) # a <= N <= b
        else:
            n_context_image = self.config.test_context
            
        n_task = self.config.n_task

        gt_ctr_ofst = torch.unsqueeze(gt_ctr_ofst, 2)
        gt_ofst = torch.cat((gt_kp_ofst, gt_ctr_ofst), dim=2)

        '''
        for i in range(bs):
            temp_cls_mask = cls_msk[i].clone()
            if temp_cls_mask.sum() < 30:
                temp_cls_mask[::4] = True

            temp_features.append(rgbd_emb[i, :, temp_cls_mask ].unsqueeze(0))
            temp_gt_ofst.append(gt_ofst[i, temp_cls_mask , :, :].unsqueeze(0))  # (bs,M,9,3)
            if cld is not None:
                temp_cld.append(cld[i, temp_cls_mask, :].unsqueeze(0)) # (bs,M,3)
            M.append(temp_gt_ofst[i].shape[1])
            temp_gt_label.append(gt_labels[i, temp_cls_mask].unsqueeze(0)) # (bs,M)
            #temp_wrong_predict.append(wrong_predict[i, cls_msk[i]])
            #temp_wrong = wrong_predict[i, cls_msk[i]]==1
            #temp_wrong = temp_wrong.view(-1)
            #print('---> Wrong prediction: ', temp_wrong.sum()/temp_wrong.shape[0])
        '''

        features_c_dict = {}  # save features of one class
        gt_ofst_c_dict = {}
        gt_label_c_dict = {}
        
        # dynamically sample the same number of context per class
        if is_training and min_num < max_num:
            random_num = torch.randint(min_num, max_num, (n_task * n_context_image,))
            temp_context_num = []
            for i in range(n_task):
                features_c_dict[i] = []
                gt_ofst_c_dict[i] = []
                gt_label_c_dict[i] = []
                temp_context_num.append(torch.sum(random_num[i::n_task]))
            M_context = int(
                sum(temp_context_num) / len(temp_context_num))  # (mean) number of seed points of all images per class
            for i in range(n_task):
                temp_lst = random_num[i::n_task]
                current_sum = torch.sum(temp_lst[1:])
                if (M_context - current_sum) < min_num or (M_context - current_sum) > max_num + 100:
                    temp_lst = torch.ones(n_context_image, dtype=torch.int16) \
                               * M_context // n_context_image
                    temp_lst[-1] = M_context - torch.sum(temp_lst[:-1])
                    random_num[i::n_task] = temp_lst
                else:
                    random_num[i] = M_context - current_sum
        else:  # sample the same number of context per image
            random_num = torch.ones(n_task * n_context_image, dtype=torch.int16) * max_num
            M_context = max_num * n_context_image
            for i in range(n_task):
                features_c_dict[i] = []
                gt_ofst_c_dict[i] = []
                gt_label_c_dict[i] = []

                #print('context = ',n_context_image, ', Random sample: ', random_num)
        
        #voted_features_c = torch.zeros(bs,M_context,temp_features[0].shape[1]) # (bs,M,128)
        voted_features_t = torch.zeros(bs,M_t, rgbd_emb.shape[1]) # (bs,M,128)
        #voted_gt_ofst_c = torch.zeros(bs,M_context,(self.n_kps + 1) * 3) # (bs,M,27)
        voted_gt_ofst_t = torch.zeros(bs, M_t, (self.n_kps + 1) * 3)  # (bs,M,27)
        voted_cld = torch.zeros(bs - n_task * n_context_image, M_t, 3) #(bs,M,3)
        voted_gt_label_t = torch.zeros(bs, M_t)  # (bs,M)
        
        #print('--------> bs = {}, n_context={}, n_task = {}, is_training ={} '.format(bs,n_context_image, n_task,is_training))
        if kps_ctr is not  None:
            batch=Batch()
            graph_list= []

        rgbd_emb = rgbd_emb.transpose(1, 2) # (bs,M,128)
        gt_ofst = gt_ofst.contiguous().view(bs, gt_ofst.shape[1] , -1)  # (bs,M,27)

        for i in range(bs):
            ## context is not a subset of target
            all_idx = [*range(M)]
            if i < n_task * n_context_image:
                #print('M[{}] = {}, random_num = {}'.format(i, M[i], random_num[i]))
                chosen_idx = random.sample(all_idx, random_num[i])
                chosen_idx.sort()
                features_c_dict[i % n_task].append(rgbd_emb[i, chosen_idx, :].unsqueeze(0))
                gt_ofst_c_dict[i % n_task].append(gt_ofst[i, chosen_idx, :].unsqueeze(0))
                gt_label_c_dict[i % n_task].append(gt_labels[i,chosen_idx].unsqueeze(0))
                #voted_cld[i, :, :] = temp_cld[i][:, chosen_idx, :]
            else:
                #random.shuffle(all_idx)
                #chosen_idx = random.sample(all_idx, M_t)
                #chosen_idx.sort()
                voted_gt_ofst_t[i,:,:] = gt_ofst[i, :, :]
                voted_features_t[i,:,:] = rgbd_emb[i, :, :]
                voted_gt_label_t[i,:] = gt_labels[i,:]
                if kps_ctr is not None:
                    temp_kps = torch.unsqueeze(kps_ctr[i], 0)
                    subgraph = BatchData(pos=temp_kps)
                    subgraph.edge_index = knn_graph(kps_ctr[i], k = self.config.graph_knn, loop=self.config.graph_knn_loop)
                    graph_list.append(subgraph)

                if cld is not None:
                    voted_cld[i - n_task * n_context_image, :, :] = cld[i, :, :]

        for i in range(n_task):
            temp_features = torch.cat(features_c_dict[i], dim=1)
            temp_gt_ofst = torch.cat(gt_ofst_c_dict[i], dim=1)
            temp_gt_label = torch.cat(gt_label_c_dict[i],dim=1)
            if i == 0:
                voted_features_c = temp_features
                voted_gt_ofst_c = temp_gt_ofst
                voted_gt_label_c = temp_gt_label
            else:
                voted_features_c = torch.cat((voted_features_c, temp_features),dim=0)
                voted_gt_ofst_c = torch.cat((voted_gt_ofst_c, temp_gt_ofst), dim=0)
                voted_gt_label_c = torch.cat((voted_gt_label_c, temp_gt_label), dim=0)

        x_context = voted_features_c[:n_task, :, :]
        y_context = voted_gt_ofst_c[:n_task, :, :]
        x_target = voted_features_t[n_task * n_context_image:, :, :]
        y_target = voted_gt_ofst_t[n_task * n_context_image:, :, :]
        label_context =  voted_gt_label_c[:n_task, :]
        label_target = voted_gt_label_t[n_task * n_context_image:, :] # (bs,M)
        label_target = torch.tensor(label_target , dtype=torch.int64)

        del temp_features
        del temp_gt_ofst
        del voted_features_c
        del voted_features_t
        del voted_gt_ofst_c
        del voted_gt_ofst_t
        del voted_gt_label_c
        del voted_gt_label_t

        if kps_ctr is None:
            if cld is None:
                return x_context, y_context, x_target, y_target, label_context, label_target
            else:
                return x_context, y_context, x_target, y_target,  label_context, label_target,voted_cld
        else:  # use GNN decoder
            full_graph = batch.from_data_list(graph_list)
            #print('Full graph: ', full_graph)
            #print("Number of sgraphs: ", full_graph.num_graphs)
            if cld is None:
                return x_context, y_context, x_target, y_target, label_context, label_target, full_graph
            else:
                return x_context, y_context, x_target, y_target,  label_context, label_target, voted_cld, full_graph


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
    inputs = load_dict("./ffb6d_inputs.pkl")

    for k,v in inputs.items():
        inputs[k] = torch.from_numpy(v)

    rgbd_emb = torch.load('./rgbd_emb.pt')
    labels = inputs['labels']
    gt_kp_ofst = inputs['kp_targ_ofst']
    gt_ctr_ofst = inputs['ctr_targ_ofst']
    cld = inputs['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()  # (bs,12800,3)

    data_split = ContextTargetSplit()
    is_train = True
    x_context, y_context, x_target, y_target, voted_cld = data_split.context_target_split(
        rgbd_emb, gt_kp_ofst, gt_ctr_ofst, labels, max_num=50, cld=cld, is_training=is_train)
    print(device)

if __name__ == "__main__":

    main()