import paddle
import paddle.nn as nn
import numpy as np

# avg pool3d
input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 2, 3, 32, 32]).astype(np.float32))
AvgPool3D = nn.AvgPool3D(kernel_size=2,
                       stride=2, padding=0)
output = AvgPool3D(input)
# output.shape [1, 2, 3, 16, 16]