import paddle
import numpy as np

x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
m = paddle.nn.Tanh()
out = m(x)
print(out)
# [-0.37994896 -0.19737532  0.09966799  0.29131261]