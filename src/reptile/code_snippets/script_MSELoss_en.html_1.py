import numpy as np
import paddle

input_data = np.array([1.5]).astype("float32")
label_data = np.array([1.7]).astype("float32")

mse_loss = paddle.nn.loss.MSELoss()
input = paddle.to_tensor(input_data)
label = paddle.to_tensor(label_data)
output = mse_loss(input, label)
print(output)
# [0.04000002]