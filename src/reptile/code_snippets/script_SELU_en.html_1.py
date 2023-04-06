import paddle
import numpy as np

x = paddle.to_tensor(np.array([[0.0, 1.0],[2.0, 3.0]]))
m = paddle.nn.SELU()
out = m(x) # [[0, 1.050701],[2.101402, 3.152103]]