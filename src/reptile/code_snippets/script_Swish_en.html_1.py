import paddle
import numpy as np

x = paddle.to_tensor(np.array([-2., 0., 1.]))
m = paddle.nn.Swish()
out = m(x) # [-0.238406, 0., 0.731059]