import paddle
import paddle.nn as nn

input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss(reduction='none')
loss = multi_label_soft_margin_loss(input, label)
print(loss)
# Tensor([3.49625897, 0.71111226, 0.43989015])

multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
loss = multi_label_soft_margin_loss(input, label)
print(loss)
# Tensor([1.54908717])