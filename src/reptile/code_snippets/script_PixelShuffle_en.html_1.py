import paddle
import paddle.nn as nn

x = paddle.randn(shape=[2,9,4,4])
pixel_shuffle = nn.PixelShuffle(3)
out_var = pixel_shuffle(x)
out = out_var.numpy()
print(out.shape)
# (2, 1, 12, 12)