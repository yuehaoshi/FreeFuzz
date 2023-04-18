import paddle
import paddle.nn as nn

input_shape = (1, 1, 2, 3)
pad = [1, 0, 1, 2]
mode = "constant"
data = paddle.arange(paddle.prod(paddle.to_tensor(input_shape)), dtype="float32").reshape(input_shape) + 1
my_pad = nn.Pad2D(padding=pad, mode=mode)
result = my_pad(data)
print(result)
# [[[[0. 0. 0. 0.]
#    [0. 1. 2. 3.]
#    [0. 4. 5. 6.]
#    [0. 0. 0. 0.]
#    [0. 0. 0. 0.]]]]