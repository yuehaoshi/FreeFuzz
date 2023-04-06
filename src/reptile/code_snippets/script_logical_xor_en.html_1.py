import paddle
import numpy as np

x_data = np.array([True, False], dtype=np.bool).reshape([2, 1])
y_data = np.array([True, False, True, False], dtype=np.bool).reshape([2, 2])
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
res = paddle.logical_xor(x, y)
print(res) # [[False,  True], [ True, False]]