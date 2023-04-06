import numpy as np
import paddle

x = paddle.to_tensor([[1, 2], [7, 8]])
y = paddle.to_tensor([[3, 4], [5, 6]])
res = paddle.maximum(x, y)
print(res)
#    [[3, 4],
#     [7, 8]]

x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
y = paddle.to_tensor([3, 0, 4])
res = paddle.maximum(x, y)
print(res)
#    [[3, 2, 4],
#     [3, 2, 4]]

x = paddle.to_tensor([2, 3, 5], dtype='float32')
y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
res = paddle.maximum(x, y)
print(res)
#    [ 2., nan, nan]

x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
res = paddle.maximum(x, y)
print(res)
#    [  5.,   3., inf.]