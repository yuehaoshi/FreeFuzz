import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
res = paddle.mean(arg_1,)
