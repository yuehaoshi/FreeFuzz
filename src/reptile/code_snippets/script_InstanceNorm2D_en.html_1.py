import paddle
import numpy as np

np.random.seed(123)
x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
x = paddle.to_tensor(x_data)
instance_norm = paddle.nn.InstanceNorm2D(2)
instance_norm_out = instance_norm(x)

print(instance_norm_out)