# adaptive avg pool2d
# suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
# output shape is [N, C, m, n], adaptive pool divide H and W dimensions
# of input data into m * n grids averagely and performs poolings in each
# grid to get output.
# adaptive avg pool performs calculations as follow:
#
#     for i in range(m):
#         for j in range(n):
#             hstart = floor(i * H / m)
#             hend = ceil((i + 1) * H / m)
#             wstart = floor(i * W / n)
#             wend = ceil((i + 1) * W / n)
#             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
#
import paddle

x = paddle.rand([2, 3, 32, 32])

adaptive_avg_pool = paddle.nn.AdaptiveAvgPool2D(output_size=3)
pool_out = adaptive_avg_pool(x = x)
# pool_out.shape is [2, 3, 3, 3]