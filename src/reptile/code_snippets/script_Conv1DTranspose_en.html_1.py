import paddle
from paddle.nn import Conv1DTranspose
import numpy as np

# shape: (1, 2, 4)
x=np.array([[[4, 0, 9, 7],
             [8, 0, 9, 2]]]).astype(np.float32)
# shape: (2, 1, 2)
y=np.array([[[7, 0]],
            [[4, 2]]]).astype(np.float32)
x_t = paddle.to_tensor(x)
conv = Conv1DTranspose(2, 1, 2)
conv.weight.set_value(y)
y_t = conv(x_t)
print(y_t)

# [[[60. 16. 99. 75.  4.]]]