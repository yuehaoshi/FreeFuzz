import paddle
import numpy as np
data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
linear = paddle.nn.Linear(32, 64)
data = paddle.to_tensor(data)
x = linear(data)
print(x.numpy())