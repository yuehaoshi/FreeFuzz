import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = True
res = paddle.cholesky(arg_1,upper=arg_2,)
