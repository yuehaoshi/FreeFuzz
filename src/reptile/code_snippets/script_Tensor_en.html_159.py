import paddle
x = paddle.ones(shape=[2, 3], dtype='int32')
x_transposed = paddle.t(x)
print(x_transposed.shape)
# [3, 2]