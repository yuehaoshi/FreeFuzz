import paddle
import paddle.nn as nn

data = paddle.uniform([1, 3, 32], dtype="float32", min=-1, max=1)
AvgPool1D = nn.AvgPool1D(kernel_size=2, stride=2, padding=0)
pool_out = AvgPool1D(data)
# pool_out shape: [1, 3, 16]