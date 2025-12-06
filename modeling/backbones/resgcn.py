import mindspore as ms
import mindspore.nn as nn
from ..modules import TemporalBasicBlock, TemporalBottleneckBlock, SpatialBasicBlock, SpatialBottleneckBlock
import logging


class ResGCNModule(nn.Cell):
    """
        ResGCNModule
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
                https://github.com/BNU-IVC/FastPoseGait
    """
    def __init__(self, in_channels, out_channels, block, A, stride=1, kernel_size=[9,2], reduction=4, get_res=False, is_main=False):
        super(ResGCNModule, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if block == 'initial':
            module_res, block_res = False, False
        elif block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: ms.Tensor(0.0)
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            # stride =2
            self.residual = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride,1), has_bias=False),
                nn.BatchNorm2d(out_channels),
            ])
        
        if block in ['Basic','initial']:
            spatial_block = SpatialBasicBlock
            temporal_block = TemporalBasicBlock
        if block == 'Bottleneck':
            spatial_block = SpatialBottleneckBlock
            temporal_block = TemporalBottleneckBlock
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, block_res, reduction)
        if in_channels == out_channels and is_main:
            tcn_stride = True
        else:
            tcn_stride = False
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res, reduction, get_res=get_res, tcn_stride=tcn_stride)
        self.edge = ms.Parameter(ms.ops.OnesLike()(A))

    def construct(self, x, A):
        A = A * self.edge
        return self.tcn(self.scn(x, A), self.residual(x))


class ResGCNInputBranch(nn.Cell):
    """
        ResGCNInputBranch_Module
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, input_branch, block, A, input_num, reduction=4):
        super(ResGCNInputBranch, self).__init__()

        # Register A as a buffer (constant in MindSpore)
        self.A = ms.Tensor(A, dtype=ms.float32)

        module_list = []
        for i in range(len(input_branch)-1):
            if i == 0:
                module_list.append(ResGCNModule(input_branch[i], input_branch[i+1], 'initial', A, reduction=reduction))
            else:
                module_list.append(ResGCNModule(input_branch[i], input_branch[i+1], block, A, reduction=reduction))

        self.bn = nn.BatchNorm2d(input_branch[0])
        self.layers = nn.CellList(module_list)

    def construct(self, x):
        x = self.bn(x)
        for layer in self.layers:
            x = layer(x, self.A)
        return x


class ResGCN(nn.Cell):
    """
        ResGCN
        Arxiv: https://arxiv.org/abs/2010.09978
    """
    def __init__(self, input_num, input_branch, main_stream, num_class, reduction, block, graph):
        super(ResGCN, self).__init__()
        self.graph = graph
        A = ms.Tensor(graph.A, dtype=ms.float32)
        
        self.head = nn.CellList([
            ResGCNInputBranch(input_branch, block, A, input_num, reduction)
            for _ in range(input_num)
        ])
        
        main_stream_list = []
        for i in range(len(main_stream)-1):
            main_stream_list.append(ResGCNModule(
                main_stream[i], main_stream[i+1], block, A, 
                stride=2 if i < len(main_stream)-2 else 1,
                is_main=(i == len(main_stream)-2),
                reduction=reduction
            ))
        self.main_stream = nn.CellList(main_stream_list)
        
        self.fc = nn.Dense(main_stream[-1], num_class)

    def construct(self, x):
        # x: [n, input_num, c, t, v]
        n, input_num, c, t, v = x.shape
        x = x.reshape(n * input_num, c, t, v)
        
        # Process each input branch
        x_list = []
        for i, head in enumerate(self.head):
            x_list.append(head(x[i::input_num]))
        stack = ops.Stack(axis=1)
        x = stack(x_list)  # [n, input_num, c, t, v]
        
        # Merge inputs
        x = x.mean(axis=1)  # [n, c, t, v]
        
        # Main stream
        for layer in self.main_stream:
            x = layer(x, self.graph.A)
        
        # Global pooling
        x = x.mean(axis=(2, 3))  # [n, c]
        x = self.fc(x)
        return x

