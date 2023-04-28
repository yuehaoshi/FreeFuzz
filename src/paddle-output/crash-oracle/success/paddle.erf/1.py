import paddle
arg_1_tensor = paddle.rand([7, 8], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.erf(arg_1,)
