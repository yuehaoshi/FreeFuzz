import numpy as np
import paddle
paddle.disable_static()

x = np.ones([2, 2], np.float32)
inputs = []
for _ in range(10):
    tmp = paddle.to_tensor(x)
    # if we don't set tmp's stop_gradient as False then, all path to loss will has no gradient since
    # there is no one need gradient on it.
    tmp.stop_gradient=False
    inputs.append(tmp)
ret = paddle.add_n(inputs)
loss = paddle.sum(ret)
loss.backward()