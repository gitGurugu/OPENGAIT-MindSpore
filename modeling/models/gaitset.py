import mindspore as ms
import copy
import mindspore.nn as nn
import mindspore.ops as ops

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv2d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper


class GaitSet(BaseModel):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
        Arxiv:  https://arxiv.org/abs/1811.06186
        Github: https://github.com/AbnerHqC/GaitSet
    """

    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        self.set_block1 = nn.SequentialCell([
            BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
            nn.LeakyReLU(alpha=0.01),
            BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
            nn.LeakyReLU(alpha=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.set_block2 = nn.SequentialCell([
            BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
            nn.LeakyReLU(alpha=0.01),
            BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
            nn.LeakyReLU(alpha=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.set_block3 = nn.SequentialCell([
            BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
            nn.LeakyReLU(alpha=0.01),
            BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
            nn.LeakyReLU(alpha=0.01)
        ])

        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(ops.ReduceMax(keep_dims=False))

        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])

        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def construct(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        if len(sils.shape) == 4:
            sils = sils.expand_dims(1)

        del ipts
        outs = self.set_block1(sils)
        gl = self.set_pooling(outs, seqL, dim=2, options={"axis": 2})[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, seqL, dim=2, options={"axis": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL, dim=2, options={"axis": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        concat = ops.Concat(axis=-1)
        feature = concat([feature1, feature2])  # [n, c, p]
        embs = self.Head(feature)

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

