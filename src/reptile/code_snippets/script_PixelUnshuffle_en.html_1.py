import paddle
import paddle.nn as nn

x = paddle.randn([2, 1, 12, 12])
pixel_unshuffle = nn.PixelUnshuffle(3)
out = pixel_unshuffle(x)
print(out.shape)
# [2, 9, 4, 4]