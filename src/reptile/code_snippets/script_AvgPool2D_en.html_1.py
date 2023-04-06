import paddle
import paddle.nn as nn
import numpy as np

# max pool2d
input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
AvgPool2D = nn.AvgPool2D(kernel_size=2,
                    stride=2, padding=0)
output = AvgPool2D(input)
# output.shape [1, 3, 16, 16]