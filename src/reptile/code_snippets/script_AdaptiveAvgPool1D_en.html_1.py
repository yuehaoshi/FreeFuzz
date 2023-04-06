# average adaptive pool1d
# suppose input data in shape of [N, C, L], `output_size` is m or [m],
# output shape is [N, C, m], adaptive pool divide L dimension
# of input data into m grids averagely and performs poolings in each
# grid to get output.
# adaptive max pool performs calculations as follow:
#
#     for i in range(m):
#         lstart = floor(i * L / m)
#         lend = ceil((i + 1) * L / m)
#         output[:, :, i] = sum(input[:, :, lstart: lend])/(lend - lstart)
#
import paddle
import paddle.nn as nn
import numpy as np

data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
AdaptiveAvgPool1D = nn.AdaptiveAvgPool1D(output_size=16)
pool_out = AdaptiveAvgPool1D(data)
# pool_out shape: [1, 3, 16]