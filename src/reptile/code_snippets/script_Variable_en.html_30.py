import paddle
import numpy as np

# x is a bool Tensor with following elements:
#    [[True, False]
#     [False, False]]
x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
print(x)
x = paddle.cast(x, 'bool')

# out1 should be [True]
out1 = paddle.any(x)  # [True]
print(out1)

# out2 should be [True, True]
out2 = paddle.any(x, axis=0)  # [True, True]
print(out2)

# keep_dim=False, out3 should be [True, True], out.shape should be (2,)
out3 = paddle.any(x, axis=-1)  # [True, True]
print(out3)

# keep_dim=True, result should be [[True], [True]], out.shape should be (2,1)
out4 = paddle.any(x, axis=1, keepdim=True)
out4 = paddle.cast(out4, 'int32')  # [[True], [True]]
print(out4)