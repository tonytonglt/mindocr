from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import XavierUniform, Constant, initializer

import numpy as np

class MeanAggregator(nn.Cell):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def construct(self, features, A):
        batmatmul = ops.BatchMatMul()
        x=batmatmul(A, features)
        return x


class GraphConv(nn.Cell):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(mindspore.Tensor(shape=(in_dim * 2, out_dim), dtype=mindspore.float32, init=XavierUniform()))
        self.bias = Parameter(mindspore.Tensor(shape=[out_dim], dtype=mindspore.float32, init=Constant(value=0)))
        self.agg = agg()

    def construct(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        agg_feats = self.agg(features, A)
        cat_feats = ops.concat((features, agg_feats), axis=2)
        # einsum = ops.Einsum("bnd,df->bnf")
        # out = einsum((cat_feats, self.weight))
        # out = ops.einsum("bnd,df->bnf", cat_feats, self.weight)
        # out = ops.einsum("ijk,kl->ijl", cat_feats, self.weight)
        # out = Tensor(np.einsum("bnd,df->bnf", cat_feats.asnumpy(), self.weight.asnumpy()))  #TODO: use ops.einsum
        out = ops.matmul(cat_feats, self.weight)
        out = nn.ReLU()(out + self.bias)
        return out


class GCN(nn.Cell):
    def __init__(self, feat_len):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(feat_len, affine=False)
        self.conv1 = GraphConv(feat_len, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 256, MeanAggregator)
        self.conv3 = GraphConv(256, 128, MeanAggregator)
        self.conv4 = GraphConv(128, 64, MeanAggregator)

        self.classifier = nn.SequentialCell(
            nn.Dense(64, 32),
            nn.PReLU(32),
            nn.Dense(32, 2))

    def construct(self, x, A, one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B, N, D = x.shape

        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        k1 = one_hop_idcs.shape[-1]
        dout = x.shape[-1]
        zeros = ops.Zeros()
        edge_feat = zeros((B, k1, dout),mindspore.float32)
        for b in range(B):
            edge_feat[b, :, :] = x[b, one_hop_idcs[b]]
        edge_feat = edge_feat.view(-1, dout)
        pred = self.classifier(edge_feat)

        # shape: (B*k1)x2
        return pred
