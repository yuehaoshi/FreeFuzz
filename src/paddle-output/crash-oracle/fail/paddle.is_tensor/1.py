import paddle
arg_1_tensor = paddle.rand([0, 3, 5], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
res = paddle.is_tensor(arg_1,)
