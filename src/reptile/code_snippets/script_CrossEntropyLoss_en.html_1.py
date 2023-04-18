# hard labels
import paddle
paddle.seed(99999)
N=100
C=200
reduction='mean'
input =  paddle.rand([N, C], dtype='float64')
label =  paddle.randint(0, C, shape=[N], dtype='int64')
weight = paddle.rand([C], dtype='float64')

cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
    weight=weight, reduction=reduction)
dy_ret = cross_entropy_loss(
                           input,
                           label)
print(dy_ret.numpy()) #[5.41993642]