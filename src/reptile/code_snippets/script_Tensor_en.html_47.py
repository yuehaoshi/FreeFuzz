import paddle
import numpy as np

x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
y_data = np.random.uniform(1, 3, [10]).astype(np.float32)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
z = paddle.dot(x, y)
print(z)