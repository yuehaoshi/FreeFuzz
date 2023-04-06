import paddle
import numpy as np

m = paddle.nn.LeakyReLU()
x = paddle.to_tensor(np.array([-2, 0, 1], 'float32'))
out = m(x)  # [-0.02, 0., 1.]