import paddle
arg_1_tensor = paddle.rand([2, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 54.0
res = paddle.pow(arg_1,arg_2,)
