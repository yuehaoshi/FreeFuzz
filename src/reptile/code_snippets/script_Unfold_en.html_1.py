import paddle
import paddle.nn as nn

x = paddle.randn((100,3,224,224))
unfold = nn.Unfold(kernel_sizes=[3, 3])
result = unfold(x)
print(result)