import numpy as np
import paddle

linear = paddle.nn.Linear(10, 3)
value = np.array([0]).astype("float32")
buffer = paddle.to_tensor(value)
linear.register_buffer("buf_name", buffer, persistable=True)

print(linear.buffers())     # == print([linear.buf_name])