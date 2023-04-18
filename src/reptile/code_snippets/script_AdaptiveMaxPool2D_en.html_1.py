# adaptive max pool2d
# suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
# output shape is [N, C, m, n], adaptive pool divide H and W dimensions
# of input data into m * n grids averagely and performs poolings in each
# grid to get output.
# adaptive max pool performs calculations as follow:
#
#     for i in range(m):
#         for j in range(n):
#             hstart = floor(i * H / m)
#             hend = ceil((i + 1) * H / m)
#             wstart = floor(i * W / n)
#             wend = ceil((i + 1) * W / n)
#             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
#
import paddle

x = paddle.rand([2, 3, 32, 32])

adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=3, return_mask=True)
pool_out, indices = adaptive_max_pool(x = x)