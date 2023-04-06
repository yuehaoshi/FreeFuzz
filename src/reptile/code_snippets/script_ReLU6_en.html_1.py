import paddle
import numpy as np

x = paddle.to_tensor(np.array([-1, 0.3, 6.5]))
m = paddle.nn.ReLU6()
out = m(x) # [0, 0.3, 6]