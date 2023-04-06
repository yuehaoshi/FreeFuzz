# adaptive max pool3d
# suppose input data in shape of [N, C, D, H, W], `output_size` is [l, m, n],
# output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
# of input data into l * m * n grids averagely and performs poolings in each
# grid to get output.
# adaptive max pool performs calculations as follow:
#
#     for i in range(l):
#         for j in range(m):
#             for k in range(n):
#                 dstart = floor(i * D / l)
#                 dend = ceil((i + 1) * D / l)
#                 hstart = floor(j * H / m)
#                 hend = ceil((j + 1) * H / m)
#                 wstart = floor(k * W / n)
#                 wend = ceil((k + 1) * W / n)
#                 output[:, :, i, j, k] =
#                     max(input[:, :, dstart:dend, hstart: hend, wstart: wend])
import paddle
import numpy as np

input_data = np.random.rand(2, 3, 8, 32, 32)
x = paddle.to_tensor(input_data)
pool = paddle.nn.AdaptiveMaxPool3D(output_size=4)
out = pool(x)
# out shape: [2, 3, 4, 4, 4]
pool = paddle.nn.AdaptiveMaxPool3D(output_size=3, return_mask=True)
out, indices = pool(x)
# out shape: [2, 3, 4, 4, 4], indices shape: [2, 3, 4, 4, 4]