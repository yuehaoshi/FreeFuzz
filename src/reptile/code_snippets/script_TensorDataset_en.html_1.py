import numpy as np
import paddle
from paddle.io import TensorDataset


input_np = np.random.random([2, 3, 4]).astype('float32')
input = paddle.to_tensor(input_np)
label_np = np.random.random([2, 1]).astype('int32')
label = paddle.to_tensor(label_np)

dataset = TensorDataset([input, label])

for i in range(len(dataset)):
    input, label = dataset[i]
    print(input, label)