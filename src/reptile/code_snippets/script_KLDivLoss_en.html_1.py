import paddle
import numpy as np
import paddle.nn as nn

shape = (5, 20)
x = np.random.uniform(-10, 10, shape).astype('float32')
target = np.random.uniform(-10, 10, shape).astype('float32')

# 'batchmean' reduction, loss shape will be [1]
kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
pred_loss = kldiv_criterion(paddle.to_tensor(x),
                            paddle.to_tensor(target))
# shape=[1]

# 'mean' reduction, loss shape will be [1]
kldiv_criterion = nn.KLDivLoss(reduction='mean')
pred_loss = kldiv_criterion(paddle.to_tensor(x),
                            paddle.to_tensor(target))
# shape=[1]

# 'sum' reduction, loss shape will be [1]
kldiv_criterion = nn.KLDivLoss(reduction='sum')
pred_loss = kldiv_criterion(paddle.to_tensor(x),
                            paddle.to_tensor(target))
# shape=[1]

# 'none' reduction, loss shape is same with X shape
kldiv_criterion = nn.KLDivLoss(reduction='none')
pred_loss = kldiv_criterion(paddle.to_tensor(x),
                            paddle.to_tensor(target))
# shape=[5, 20]