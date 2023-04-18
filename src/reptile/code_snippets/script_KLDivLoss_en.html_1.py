import paddle
import paddle.nn as nn

shape = (5, 20)
x = paddle.uniform(shape, min=-10, max=10).astype('float32')
target = paddle.uniform(shape, min=-10, max=10).astype('float32')

# 'batchmean' reduction, loss shape will be [1]
kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
pred_loss = kldiv_criterion(x, target)
# shape=[1]

# 'mean' reduction, loss shape will be [1]
kldiv_criterion = nn.KLDivLoss(reduction='mean')
pred_loss = kldiv_criterion(x, target)
# shape=[1]

# 'sum' reduction, loss shape will be [1]
kldiv_criterion = nn.KLDivLoss(reduction='sum')
pred_loss = kldiv_criterion(x, target)
# shape=[1]

# 'none' reduction, loss shape is same with X shape
kldiv_criterion = nn.KLDivLoss(reduction='none')
pred_loss = kldiv_criterion(x, target)
# shape=[5, 20]