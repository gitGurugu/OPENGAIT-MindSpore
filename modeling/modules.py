import mindspore as ms
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from utils import clones, is_list_or_tuple
from einops import rearrange


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.shape[:2]
        features = []
        for b in self.bin_num:
            z = x.reshape(n, c, b, -1)
            z = z.mean(axis=-1) + z.max(axis=-1)[0]
            features.append(z)
        concat = ops.Concat(axis=-1)
        return concat(features)


class SetBlockWrapper(nn.Cell):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def construct(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.shape
        x = x.transpose(1, 2).reshape(-1, c, h, w)
        x = self.forward_block(x, *args, **kwargs)
        output_size = x.shape
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2)


class PackSequenceWrapper(nn.Cell):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def construct(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].asnumpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs[:, :, curr_start:curr_start+curr_seqL]
            rets.append(self.pooling_func(narrowed_seq, **options))
        concat = ops.Concat(axis=0)
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [concat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return concat(rets)


class BasicConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, pad_mode='pad', padding=padding, has_bias=False, **kwargs)

    def construct(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Cell):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        # Initialize parameter
        self.fc_bin = ms.Parameter(
            ms.common.initializer.initializer(
                ms.common.initializer.XavierUniform(),
                (parts_num, in_channels, out_channels),
                ms.float32
            )
        )
        self.norm = norm
        self.l2norm = ops.L2Normalize(axis=1)

    def construct(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.transpose(2, 0, 1)  # [p, n, c_in]
        if self.norm:
            out = ops.BatchMatMul()(x, self.l2norm(self.fc_bin))
        else:
            out = ops.BatchMatMul()(x, self.fc_bin)
        return out.transpose(1, 2, 0)  # [n, c_out, p]


class SeparateBNNecks(nn.Cell):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = ms.Parameter(
            ms.common.initializer.initializer(
                ms.common.initializer.XavierUniform(),
                (parts_num, in_channels, class_num),
                ms.float32
            )
        )
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d
        self.l2norm = ops.L2Normalize(axis=-1)
        self.split = ops.Split(axis=2, output_num=parts_num)

    def construct(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.shape
            x = x.reshape(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.reshape(n, c, p)
        else:
            x_list = self.split(x)  # List of [n, c, 1]
            x_list = [bn(x_list[i].squeeze(2)) for i, bn in enumerate(self.bn1d)]
            stack = ops.Stack(axis=2)
            x = stack(x_list)  # [n, c, p]
        feature = x.transpose(2, 0, 1)  # [p, n, c]
        if self.norm:
            feature = self.l2norm(feature)  # [p, n, c]
            logits = ops.BatchMatMul()(feature, self.l2norm(self.fc_bin))  # [p, n, c]
        else:
            logits = ops.BatchMatMul()(feature, self.fc_bin)
        return feature.transpose(1, 2, 0), logits.transpose(1, 2, 0)  # [n, c, p]


class FocalConv2d(nn.Cell):
    """
        GaitPart: Temporal Part-based Model for Gait Recognition
        CVPR2020: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
        Github: https://github.com/ChaoFan96/GaitPart
    """
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, has_bias=False, pad_mode='pad', **kwargs)
        self.split = ops.Split(axis=2)

    def construct(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.shape[2]
            split_size = int(h // 2**self.halving)
            # Split along height dimension
            z_list = []
            for i in range(0, h, split_size):
                end = min(i + split_size, h)
                z_list.append(self.conv(x[:, :, i:end, :]))
            concat = ops.Concat(axis=2)
            z = concat(z_list)
        return z


class BasicConv3d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, pad_mode='pad', padding=padding, has_bias=bias, **kwargs)

    def construct(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


# Note: GaitAlign uses RoIAlign which may need special implementation in MindSpore
# For now, we provide a simplified version
class GaitAlign(nn.Cell):
    """
        GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality
        ECCV2022: https://arxiv.org/pdf/2203.03972v2.pdf
        Github: https://github.com/ShiqiYu/OpenGait/tree/master/configs/gaitedge
    """
    def __init__(self, H=64, W=44, eps=1, **kwargs):
        super(GaitAlign, self).__init__()
        self.H, self.W, self.eps = H, W, eps
        self.Pad = nn.Pad(((0, 0), (0, 0), (0, 0), (int(self.W / 2), int(self.W / 2))))
        # Note: MindSpore doesn't have RoIAlign built-in, may need custom implementation
        # For now, using a placeholder
        self.RoiPool = None  # TODO: Implement RoIAlign for MindSpore

    def construct(self, feature_map, binary_mask, w_h_ratio):
        """
           In  sils:         [n, c, h, w]
               w_h_ratio:    [n, 1]
           Out aligned_sils: [n, c, H, W]
        """
        # Simplified version - full implementation requires RoIAlign
        # This is a placeholder
        n, c, h, w = feature_map.shape
        w_h_ratio = w_h_ratio.reshape(-1, 1)  # [n, 1]

        h_sum = binary_mask.sum(axis=-1)  # [n, c, h]
        _ = (h_sum >= self.eps).astype(ms.float32).cumsum(axis=-1)  # [n, c, h]
        h_top = (_ == 0).astype(ms.float32).sum(axis=-1)  # [n, c]
        reduce_max = ops.ReduceMax(keep_dims=True)
        h_bot = (_ != reduce_max(_, axis=-1)).astype(ms.float32).sum(axis=-1) + 1.  # [n, c]

        w_sum = binary_mask.sum(axis=-2)  # [n, c, w]
        w_cumsum = w_sum.cumsum(axis=-1)  # [n, c, w]
        w_h_sum = w_sum.sum(axis=-1, keepdims=True)  # [n, c, 1]
        w_center = (w_cumsum < w_h_sum / 2.).astype(ms.float32).sum(axis=-1)  # [n, c]

        p1 = self.W - self.H * w_h_ratio
        p1 = p1 / 2.
        p1 = ops.Maximum()(p1, ms.Tensor(0.0))  # [n, c]
        t_w = w_h_ratio * self.H / w
        p2 = p1 / t_w  # [n, c]

        height = h_bot - h_top  # [n, c]
        width = height * w / h  # [n, c]
        width_p = int(self.W / 2)

        feature_map = self.Pad(feature_map)
        w_center = w_center + width_p  # [n, c]

        w_left = w_center - width / 2 - p2  # [n, c]
        w_right = w_center + width / 2 + p2  # [n, c]

        w_left = ops.ClipByValue()(w_left, ms.Tensor(0.0), ms.Tensor(float(w+2*width_p)))
        w_right = ops.ClipByValue()(w_right, ms.Tensor(0.0), ms.Tensor(float(w+2*width_p)))

        # TODO: Implement proper RoIAlign
        # For now, return a resized version
        from mindspore.ops import ResizeBilinear
        resize = ResizeBilinear()
        crops = resize(feature_map, (self.H, self.W))
        return crops


def RmBN2dAffine(model):
    for m in model.cells():
        if isinstance(m, nn.BatchNorm2d):
            m.gamma.requires_grad = False
            m.beta.requires_grad = False


'''
Modifed from https://github.com/BNU-IVC/FastPoseGait/blob/main/fastposegait/modeling/components/units
'''

class Graph():
    """
    # Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
    """
    def __init__(self, joint_format='coco', max_hop=2, dilation=1):
        self.joint_format = joint_format
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return str(self.A)

    def _get_edge(self):
        if self.joint_format == 'coco':
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            parts = [
                np.array([5, 7, 9]),                      # left_arm
                np.array([6, 8, 10]),                     # right_arm
                np.array([11, 13, 15]),                   # left_leg
                np.array([12, 14, 16]),                   # right_leg
                np.array([0, 1, 2, 3, 4]),                # head
            ]

        elif self.joint_format == 'coco-no-head':
            num_node = 12
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1),
                             (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
                             (6, 8), (8, 10), (7, 9), (9, 11)]
            self.edge = self_link + neighbor_link
            self.center = 0
            connect_joint = np.array([3,1,0,2,4,0,6,8,10,7,9,11])
            parts =[
                np.array([0, 2, 4]),       # left_arm
                np.array([1, 3, 5]),       # right_arm
                np.array([6, 8, 10]),      # left_leg
                np.array([7, 9, 11])       # right_leg
            ]

        elif self.joint_format =='alphapose' or self.joint_format =='openpose':
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.flip_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),               # left_arm
                np.array([2, 3, 4]),               # right_arm
                np.array([11, 12, 13]),            # left_leg
                np.array([8, 9, 10]),              # right_leg
                np.array([0, 1, 14, 15, 16, 17]),  # head
            ]

        else:
            num_node, neighbor_link, connect_joint, parts = 0, [], [], []
            raise ValueError('Error: Do NOT exist this joint format: {}!'.format(self.joint_format))
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD


class TemporalBasicBlock(nn.Cell):
    """
        TemporalConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, channels, temporal_window_size, stride=1, residual=False,reduction=0,get_res=False,tcn_stride=False):
        super(TemporalBasicBlock, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: ms.Tensor(0.0)
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.SequentialCell([
                nn.Conv2d(channels, channels, 1, stride=(stride,1), pad_mode='pad', padding=0, has_bias=False),
                nn.BatchNorm2d(channels),
            ])

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), stride=(stride,1), 
                              pad_mode='pad', padding=padding, has_bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def construct(self, x, res_module):
        res_block = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)
        return x


class TemporalBottleneckBlock(nn.Cell):
    """
        TemporalConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=4,get_res=False, tcn_stride=False):
        super(TemporalBottleneckBlock, self).__init__()
        tcn_stride = False
        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // reduction
        if get_res:
            if tcn_stride:
                stride = 2
            self.residual = nn.SequentialCell([
                nn.Conv2d(channels, channels, 1, stride=(2,1), pad_mode='pad', padding=0, has_bias=False),
                nn.BatchNorm2d(channels),
            ])
            tcn_stride = True
        else:
            if not residual:
                self.residual = lambda x: ms.Tensor(0.0)
            elif stride == 1:
                self.residual = lambda x: x
            else:
                self.residual = nn.SequentialCell([
                    nn.Conv2d(channels, channels, 1, stride=(2,1), pad_mode='pad', padding=0, has_bias=False),
                    nn.BatchNorm2d(channels),
                ])
                tcn_stride = True

        self.conv_down = nn.Conv2d(channels, inter_channels, 1, has_bias=False)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        if tcn_stride:
            stride = 2
        self.conv = nn.Conv2d(inter_channels, inter_channels, (temporal_window_size,1), stride=(stride,1), 
                              pad_mode='pad', padding=padding, has_bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, channels, 1, has_bias=False)
        self.bn_up = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def construct(self, x, res_module):
        res_block = self.residual(x)
        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block + res_module)
        return x


class SpatialGraphConv(nn.Cell):
    """
        SpatialGraphConv_Basic_Block
        Arxiv: https://arxiv.org/abs/1801.07455
        Github: https://github.com/yysijie/st-gcn
    """
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1, has_bias=False)

    def construct(self, x, A):
        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.shape
        x = x.reshape(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        # spatial graph convolution
        # x: [n, s_kernel_size, kc//s_kernel_size, t, v]
        # A: [s_kernel_size, v, w]
        # Use einsum for efficient computation
        x = ops.Einsum('nkctv,kvw->nctw')((x, A[:self.s_kernel_size]))

        return x


class SpatialBasicBlock(nn.Cell):
    """
        SpatialGraphConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False,reduction=0):
        super(SpatialBasicBlock, self).__init__()

        if not residual:
            self.residual = lambda x: ms.Tensor(0.0)
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, 1, has_bias=False),
                nn.BatchNorm2d(out_channels),
            ])

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x, A):
        res_block = self.residual(x)
        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x + res_block)
        return x


class SpatialBottleneckBlock(nn.Cell):
    """
        SpatialGraphConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, reduction=4):
        super(SpatialBottleneckBlock, self).__init__()

        inter_channels = out_channels // reduction

        if not residual:
            self.residual = lambda x: ms.Tensor(0.0)
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, 1, has_bias=False),
                nn.BatchNorm2d(out_channels),
            ])

        self.conv_down = nn.Conv2d(in_channels, inter_channels, 1, has_bias=False)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        self.conv = SpatialGraphConv(inter_channels, inter_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, out_channels, 1, has_bias=False)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x, A):
        res_block = self.residual(x)
        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)
        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block)
        return x


# Additional modules for skeleton-based methods
class ParallelBN1d(nn.Cell):
    def __init__(self, parts_num, in_channels, **kwargs):
        super(ParallelBN1d, self).__init__()
        self.parts_num = parts_num
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num, **kwargs)

    def construct(self, x):
        '''
            x: [n, c, p]
        '''
        x = rearrange(x, 'n c p -> n (c p)')
        x = self.bn1d(x)
        x = rearrange(x, 'n (c p) -> n c p', p=self.parts_num)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     pad_mode='pad', padding=dilation, group=groups, has_bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False)


class BasicBlock2D(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

