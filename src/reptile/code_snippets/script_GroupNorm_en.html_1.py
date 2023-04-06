import paddle
import numpy as np

paddle.disable_static()
np.random.seed(123)
x_data = np.random.random(size=(2, 6, 2, 2)).astype('float32')
x = paddle.to_tensor(x_data)
group_norm = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
group_norm_out = group_norm(x)

print(group_norm_out.numpy())