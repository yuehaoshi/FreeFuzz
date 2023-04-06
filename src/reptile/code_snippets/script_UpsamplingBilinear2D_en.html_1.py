import paddle
import paddle.nn as nn

input_data = paddle.rand(shape=(2,3,6,10)).astype("float32")
upsample_out  = paddle.nn.UpsamplingBilinear2D(size=[12,12])
input = paddle.to_tensor(input_data)
output = upsample_out(x=input)
print(output.shape)
# [2L, 3L, 12L, 12L]