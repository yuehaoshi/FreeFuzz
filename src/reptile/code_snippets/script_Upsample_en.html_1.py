import paddle
import paddle.nn as nn
import numpy as np

input_data = np.random.rand(2,3,6,10).astype("float32")
upsample_out  = paddle.nn.Upsample(size=[12,12])

input = paddle.to_tensor(input_data)
output = upsample_out(x=input)
print(output.shape)
# [2L, 3L, 12L, 12L]