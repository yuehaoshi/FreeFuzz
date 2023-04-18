import paddle
arg_1_tensor = paddle.rand([0, 0], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.imag(arg_1,)
