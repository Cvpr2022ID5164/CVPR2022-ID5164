import torch
from torch_geometric.nn import PointConv
from  torch_geometric.data import Data
from torch_sparse import SparseTensor
import re


class BatchData(Data):
    '''
    Handle batch data (M, num_nodes, c)
    '''
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.normal = normal
        self.face = face
        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and edge_index.dtype != torch.long:
            raise ValueError(
                (f'Argument `edge_index` needs to be of type `torch.long` but '
                 f'found type `{edge_index.dtype}`.'))

        if face is not None and face.dtype != torch.long:
            raise ValueError(
                (f'Argument `face` needs to be of type `torch.long` but found '
                 f'type `{face.dtype}`.'))

    def __cat_dim__(self, key, value,*args, **kwargs):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # By default, concatenate sparse matrices diagonally.
        if isinstance(value, SparseTensor):
            return (0, 1)
        # Concatenate `*index*` and `*face*` attributes in the last dimension.
        elif bool(re.search('(index|face)', key)):
            return -1
        elif key == 'batch':
            return 0
        return 1


class BatchPointConv(PointConv):
    '''
    Handle batch data (M, num_nodes, c)
    '''
    def __init__(self, local_nn=None,
                 global_nn=None,
                 add_self_loops=True, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def message(self, x_j, pos_i, pos_j):
        msg = pos_j - pos_i

        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=-1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

