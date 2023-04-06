import paddle
import numpy as np

np.random.seed(123)
x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
x = paddle.to_tensor(x_data)
layer_norm = paddle.nn.LayerNorm(x_data.shape[1:])
layer_norm_out = layer_norm(x)

print(layer_norm_out)