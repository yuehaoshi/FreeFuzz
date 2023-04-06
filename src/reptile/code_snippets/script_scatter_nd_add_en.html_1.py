import paddle
import numpy as np

x = paddle.rand(shape=[3, 5, 9, 10], dtype='float32')
updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
index_data = np.array([[1, 1],
                       [0, 1],
                       [1, 3]]).astype(np.int64)
index = paddle.to_tensor(index_data)
output = paddle.scatter_nd_add(x, index, updates)