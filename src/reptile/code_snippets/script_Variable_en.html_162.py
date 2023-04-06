import numpy as np
import paddle

x = paddle.to_tensor([[1, 2], [7, 8]])
y = paddle.to_tensor([[5, 6], [3, 4]])
res = paddle.subtract(x, y)
print(res)
#       [[-4, -4],
#        [4, 4]]

x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
y = paddle.to_tensor([1, 0, 4])
res = paddle.subtract(x, y)
print(res)
#       [[[ 0,  2, -1],
#         [ 0,  2, -1]]]

x = paddle.to_tensor([2, np.nan, 5], dtype='float32')
y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
res = paddle.subtract(x, y)
print(res)
#       [ 1., nan, nan]

x = paddle.to_tensor([5, np.inf, -np.inf], dtype='float64')
y = paddle.to_tensor([1, 4, 5], dtype='float64')
res = paddle.subtract(x, y)
print(res)
#       [   4.,  inf., -inf.]