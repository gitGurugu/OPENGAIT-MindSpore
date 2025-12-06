import mindspore as ms
from mindspore import nn, Parameter
import mindspore.ops as ops
import math


class Normalize(nn.Cell):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def construct(self, x):
        norm = (x ** self.power).sum(axis=1, keepdims=True) ** (1. / self.power)
        out = x / norm
        return out


class GraphConvolution(nn.Cell):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_size=9, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_size = adj_size

        self.weight = Parameter(ms.common.initializer.initializer(
            ms.common.initializer.Uniform(1.0 / math.sqrt(in_features)),
            (in_features, out_features),
            ms.float32
        ))
        
        if bias:
            self.bias = Parameter(ms.common.initializer.initializer(
                ms.common.initializer.Uniform(1.0 / math.sqrt(in_features)),
                (out_features,),
                ms.float32
            ))
        else:
            self.bias = None
        self.bn = nn.BatchNorm1d(out_features * adj_size)

    def construct(self, input, adj):
        support = ops.MatMul()(input, self.weight)
        output_ = ops.BatchMatMul()(adj, support)
        if self.bias is not None:
            output_ = output_ + self.bias
        output = output_.reshape(output_.shape[0], output_.shape[1]*output_.shape[2])
        output = self.bn(output)
        output = output.reshape(output_.shape[0], output_.shape[1], output_.shape[2])

        return output


class GCN(nn.Cell):
    def __init__(self, adj_size, nfeat, nhid, isMeanPooling=True):
        super(GCN, self).__init__()

        self.adj_size = adj_size
        self.nhid = nhid
        self.isMeanPooling = isMeanPooling
        self.gc1 = GraphConvolution(nfeat, nhid, adj_size)
        self.gc2 = GraphConvolution(nhid, nhid, adj_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def construct(self, x, adj):
        x_ = self.dropout(x)
        x_ = self.relu(self.gc1(x_, adj))
        x_ = self.dropout(x_)
        x_ = self.relu(self.gc2(x_, adj))
        return x_

