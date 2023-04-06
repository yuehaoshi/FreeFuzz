import paddle
import paddle.nn as nn
import numpy as np

x = np.random.randn(2, 9, 4, 4).astype(np.float32)
x_var = paddle.to_tensor(x)
pixel_shuffle = nn.PixelShuffle(3)
out_var = pixel_shuffle(x_var)
out = out_var.numpy()
print(out.shape)
# (2, 1, 12, 12)