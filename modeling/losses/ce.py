import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy
        self.softmax_cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        if label_smooth:
            self.softmax_cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')

    def construct(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.shape
        logits = logits.astype(ms.float32)
        labels = labels.expand_dims(1)  # [n, 1]
        
        # Reshape for cross entropy: [n, c, p] -> [n*p, c]
        logits_reshaped = logits.transpose(0, 2, 1).reshape(-1, c)  # [n*p, c]
        logits_scaled = logits_reshaped * self.scale
        
        if self.label_smooth:
            # Create one-hot labels with smoothing
            labels_one_hot = ops.OneHot()(labels.repeat(1, p).reshape(-1), c, 1.0 - self.eps, self.eps / (c - 1))
            loss = self.softmax_cross_entropy(logits_scaled, labels_one_hot)
        else:
            labels_reshaped = labels.repeat(1, p).reshape(-1)  # [n*p]
            loss = self.softmax_cross_entropy(logits_scaled, labels_reshaped)
        
        self.info.update({'loss': loss.asnumpy()})
        if self.log_accuracy:
            pred = logits.argmax(axis=1)  # [n, p]
            accu = (pred == labels).astype(ms.float32).mean()
            self.info.update({'accuracy': accu.asnumpy()})
        return loss, self.info

