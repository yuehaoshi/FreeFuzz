import paddle
import numpy as np

x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
m = paddle.nn.Tanhshrink()
out = m(x) # [-0.020051, -0.00262468, 0.000332005, 0.00868739]