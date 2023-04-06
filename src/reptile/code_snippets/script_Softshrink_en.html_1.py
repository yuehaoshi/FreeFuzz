import paddle
import numpy as np

x = paddle.to_tensor(np.array([-0.9, -0.2, 0.1, 0.8]))
m = paddle.nn.Softshrink()
out = m(x) # [-0.4, 0, 0, 0.3]