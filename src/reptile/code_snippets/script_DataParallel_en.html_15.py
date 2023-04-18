import numpy as np
import paddle

fc1 = paddle.nn.Linear(10, 3)
buffer1 = paddle.to_tensor(np.array([0]).astype("float32"))
# register a tensor as buffer by specific `persistable`
fc1.register_buffer("buf_name_1", buffer1, persistable=True)

fc2 = paddle.nn.Linear(3, 10)
buffer2 = paddle.to_tensor(np.array([1]).astype("float32"))
# register a buffer by assigning an attribute with Tensor.
# The `persistable` can only be False by this way.
fc2.buf_name_2 = buffer2

model = paddle.nn.Sequential(fc1, fc2)

# get all named buffers
for name, buffer in model.named_buffers():
    print(name, buffer)