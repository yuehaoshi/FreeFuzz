import numpy as np
import paddle

# x is a Tensor which shape is [3, 9, 5]
x_np = np.random.random([3, 9, 5]).astype("int32")
x = paddle.to_tensor(x_np)

out0, out1, out2 = paddle.chunk(x, chunks=3, axis=1)
# out0.shape [3, 3, 5]
# out1.shape [3, 3, 5]
# out2.shape [3, 3, 5]


# axis is negative, the real axis is (rank(x) + axis) which real
# value is 1.
out0, out1, out2 = paddle.chunk(x, chunks=3, axis=-2)
# out0.shape [3, 3, 5]
# out1.shape [3, 3, 5]
# out2.shape [3, 3, 5]