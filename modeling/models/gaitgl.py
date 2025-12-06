import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks


class GLConv(nn.Cell):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.leaky_relu = nn.LeakyReLU(alpha=0.01)

    def construct(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.shape[3]
            split_size = int(h // 2**self.halving)
            # Split along height dimension
            lcl_feat_list = []
            for i in range(0, h, split_size):
                end = min(i + split_size, h)
                lcl_feat_list.append(self.local_conv3d(x[:, :, :, i:end, :]))
            concat = ops.Concat(axis=3)
            lcl_feat = concat(lcl_feat_list)

        if not self.fm_sign:
            feat = self.leaky_relu(gob_feat) + self.leaky_relu(lcl_feat)
        else:
            concat = ops.Concat(axis=3)
            feat = self.leaky_relu(concat([gob_feat, lcl_feat]))
        return feat


class GeMHPP(nn.Cell):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = ms.Parameter(ms.Tensor([p], dtype=ms.float32))
        self.eps = eps
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(1, 1), stride=1)

    def gem(self, ipts):
        # ipts: [n, c, h, w]
        clamped = ops.Maximum()(ipts, ms.Tensor(self.eps))
        powered = ops.Pow()(clamped, self.p)
        pooled = self.avg_pool2d(powered)
        result = ops.Pow()(pooled, 1.0 / self.p)
        return result

    def construct(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.shape[:2]
        features = []
        for b in self.bin_num:
            z = x.reshape(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        concat = ops.Concat(axis=-1)
        return concat(features)


class GaitGL(BaseModel):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, *args, **kargs):
        super(GaitGL, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']

        if dataset_name in ['OUMVLP', 'GREW']:
            # For OUMVLP and GREW
            self.conv3d = nn.SequentialCell([
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(alpha=0.01),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(alpha=0.01),
            ])
            self.LTA = nn.SequentialCell([
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(alpha=0.01)
            ])

            self.GLConvA0 = nn.SequentialCell([
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ])
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.SequentialCell([
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ])
            self.GLConvB2 = nn.SequentialCell([
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ])
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.SequentialCell([
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(alpha=0.01)
            ])
            self.LTA = nn.SequentialCell([
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(alpha=0.01)
            ])

            self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.TP = PackSequenceWrapper(ops.ReduceMax(keep_dims=False))
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

    def construct(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].expand_dims(1)
        del ipts
        n, _, s, h, w = sils.shape
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = ops.Tile()(sils, (1, 1, repeat, 1, 1))

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, dim=2, options={"axis": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        n, _, s, h, w = sils.shape
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.reshape(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval

