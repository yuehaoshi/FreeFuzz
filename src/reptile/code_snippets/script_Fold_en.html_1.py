import paddle
import paddle.nn as nn

x = paddle.randn([2,3*2*2,12])
fold = nn.Fold(output_sizes=[4, 5], kernel_sizes=2)
y = fold(x)
# y.shape = [2,3,4,5]