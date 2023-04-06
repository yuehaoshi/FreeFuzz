import paddle
from paddle.nn import Conv1D
import numpy as np
x = np.array([[[4, 8, 1, 9],
  [7, 2, 0, 9],
  [6, 9, 2, 6]]]).astype(np.float32)
w=np.array(
[[[9, 3, 4],
  [0, 0, 7],
  [2, 5, 6]],
 [[0, 3, 4],
  [2, 9, 7],
  [5, 6, 8]]]).astype(np.float32)
x_t = paddle.to_tensor(x)
conv = Conv1D(3, 2, 3)
conv.weight.set_value(w)
y_t = conv(x_t)
print(y_t)
# [[[133. 238.]
#   [160. 211.]]]