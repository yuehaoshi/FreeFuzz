import paddle
import numpy as np
paddle.disable_static()
x_np = np.array([[1., 3.], [3., 5.]]).astype(np.float64)
y_np = np.array([[5., 6.], [7., 8.]]).astype(np.float64)
x = paddle.to_tensor(x_np)
y = paddle.to_tensor(y_np)
dist = paddle.nn.PairwiseDistance()
distance = dist(x, y)
print(distance.numpy()) # [5. 5.]