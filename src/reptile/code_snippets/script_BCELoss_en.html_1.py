import numpy as np
import paddle
input_data = np.array([0.5, 0.6, 0.7]).astype("float32")
label_data = np.array([1.0, 0.0, 1.0]).astype("float32")

input = paddle.to_tensor(input_data)
label = paddle.to_tensor(label_data)
bce_loss = paddle.nn.BCELoss()
output = bce_loss(input, label)
print(output)  # [0.65537095]