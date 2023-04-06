import paddle
import numpy as np
input_data = np.random.rand(3,3).astype("float32")
label_data = np.random.rand(3,3).astype("float32")
input = paddle.to_tensor(input_data)
label = paddle.to_tensor(label_data)
loss = paddle.nn.SmoothL1Loss()
output = loss(input, label)
print(output)