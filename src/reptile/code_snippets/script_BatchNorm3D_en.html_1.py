import paddle
import numpy as np

np.random.seed(123)
x_data = np.random.random(size=(2, 1, 2, 2, 3)).astype('float32')
x = paddle.to_tensor(x_data)
batch_norm = paddle.nn.BatchNorm3D(1)
batch_norm_out = batch_norm(x)

print(batch_norm_out)