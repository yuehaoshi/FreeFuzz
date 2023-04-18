import paddle
import paddle.nn as nn

# avg pool3d
input = paddle.uniform([1, 2, 3, 32, 32], dtype="float32", min=-1, max=1)
AvgPool3D = nn.AvgPool3D(kernel_size=2,
                       stride=2, padding=0)
output = AvgPool3D(input)
# output.shape [1, 2, 3, 16, 16]