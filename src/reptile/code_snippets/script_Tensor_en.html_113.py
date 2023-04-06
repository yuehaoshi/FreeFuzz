# x: [M, N], vec: [N]
# paddle.mv(x, vec)  # out: [M]

import numpy as np
import paddle

x_data = np.array([[2, 1, 3], [3, 0, 1]]).astype("float64")
x = paddle.to_tensor(x_data)
vec_data = np.array([3, 5, 1])
vec = paddle.to_tensor(vec_data).astype("float64")
out = paddle.mv(x, vec)