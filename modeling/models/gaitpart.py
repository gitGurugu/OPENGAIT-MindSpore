import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs
from utils import clones


class BasicConv1d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, has_bias=False, pad_mode='pad', **kwargs)

    def construct(self, x):
        ret = self.conv(x)
        return ret


class TemporalFeatureAggregator(nn.Cell):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.SequentialCell([
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(alpha=0.01),
            BasicConv1d(hidden_dim, in_channels, 1)])
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.max_pool3x1 = nn.MaxPool1d(kernel_size=3, stride=1, pad_mode='pad', padding=1)

        # MTB2
        conv3x3 = nn.SequentialCell([
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(alpha=0.01),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1)])
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(kernel_size=5, stride=1, pad_mode='pad', padding=2)
        self.max_pool3x3 = nn.MaxPool1d(kernel_size=5, stride=1, pad_mode='pad', padding=2)

        # Temporal Pooling, TP
        self.TP = ops.ReduceMax(keep_dims=False)
        self.sigmoid = ops.Sigmoid()
        self.split = ops.Split(axis=0, output_num=parts_num)

    def construct(self, x):
        """
          Input:  x,   [n, c, s, p]
          Output: ret, [n, c, p]
        """
        n, c, s, p = x.shape
        x = x.transpose(3, 0, 1, 2)  # [p, n, c, s]
        feature_list = self.split(x)  # List of [1, n, c, s]
        x = x.reshape(-1, c, s)  # [p, n, c, s] -> [p*n, c, s]

        # MTB1: ConvNet1d & Sigmoid
        logits3x1_list = []
        for i, (conv, feat) in enumerate(zip(self.conv1d3x1, feature_list)):
            logits3x1_list.append(conv(feat.squeeze(0)).expand_dims(0))
        concat = ops.Concat(axis=0)
        logits3x1 = concat(logits3x1_list)
        scores3x1 = self.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.reshape(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3_list = []
        for i, (conv, feat) in enumerate(zip(self.conv1d3x3, feature_list)):
            logits3x3_list.append(conv(feat.squeeze(0)).expand_dims(0))
        concat = ops.Concat(axis=0)
        logits3x3 = concat(logits3x3_list)
        scores3x3 = self.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.reshape(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, axis=-1)  # [p, n, c]
        ret = ret.transpose(1, 2, 0)  # [n, c, p]
        return ret


class GaitPart(BaseModel):
    def __init__(self, *args, **kargs):
        super(GaitPart, self).__init__(*args, **kargs)
        """
            GaitPart: Temporal Part-based Model for Gait Recognition
            Paper:    https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
            Github:   https://github.com/ChaoFan96/GaitPart
        """

    def build_network(self, model_cfg):

        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        head_cfg = model_cfg['SeparateFCs']
        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.HPP = SetBlockWrapper(
            HorizontalPoolingPyramid(bin_num=model_cfg['bin_num']))
        self.TFA = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=head_cfg['in_channels'], parts_num=head_cfg['parts_num']))

    def construct(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.shape) == 4:
            sils = sils.expand_dims(1)

        del ipts
        out = self.Backbone(sils)  # [n, c, s, h, w]
        out = self.HPP(out)  # [n, c, s, p]
        out = self.TFA(out, seqL)  # [n, c, p]

        embs = self.Head(out)  # [n, c, p]

        n, _, s, h, w = sils.shape
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.reshape(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval

