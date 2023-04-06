import paddle
import numpy as np

x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
m = paddle.nn.Softsign()
out = m(x) # [-0.285714, -0.166667, 0.0909091, 0.230769]