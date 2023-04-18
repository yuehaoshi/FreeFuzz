import paddle
import paddle.nn as nn

# max pool2d
input = paddle.uniform([1, 3, 32, 32], dtype="float32", min=-1, max=1)
AvgPool2D = nn.AvgPool2D(kernel_size=2,
                    stride=2, padding=0)
output = AvgPool2D(input)
# output.shape [1, 3, 16, 16]