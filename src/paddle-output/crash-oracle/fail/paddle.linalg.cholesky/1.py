import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -63.0
res = paddle.linalg.cholesky(arg_1,upper=arg_2,)
