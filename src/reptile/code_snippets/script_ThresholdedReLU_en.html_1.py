import paddle
import numpy as np

x = paddle.to_tensor(np.array([2., 0., 1.]))
m = paddle.nn.ThresholdedReLU()
out = m(x) # [2., 0., 0.]