import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.relu = ops.ReLU()
        self.sqrt = ops.Sqrt()

    @gather_and_scale_wrapper
    def construct(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.transpose(2, 0, 1).astype(ms.float32)  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean(axis=(1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = (ap_dist - an_dist).reshape(dist.shape[0], -1)
        loss = self.relu(dist_diff + self.margin)

        hard_loss = loss.max(axis=-1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.asnumpy(),
            'hard_loss': hard_loss.asnumpy(),
            'loss_num': loss_num.asnumpy(),
            'mean_dist': mean_dist.asnumpy()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(axis=-1)
        loss_num = (loss != 0).sum(axis=-1).astype(ms.float32)

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg = ops.select(loss_num == 0, 
                             ops.zeros_like(loss_avg), loss_avg)
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = (x ** 2).sum(axis=-1, keepdims=True)  # [p, n_x, 1]
        y2 = (y ** 2).sum(axis=-1, keepdims=True)  # [p, 1, n_y]
        y2 = y2.transpose(0, 2, 1)  # [p, n_y, 1] -> [p, 1, n_y]
        inner = ops.BatchMatMul()(x, y.transpose(0, 2, 1))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = self.sqrt(self.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.expand_dims(1) ==
                   clo_label.expand_dims(0)).astype(ms.bool_)  # [n_r, n_c]
        diffenc = ops.LogicalNot()(matches)  # [n_r, n_c]
        p, n, _ = dist.shape
        ap_dist = dist[:, matches].reshape(p, n, -1, 1)
        an_dist = dist[:, diffenc].reshape(p, n, 1, -1)
        return ap_dist, an_dist

