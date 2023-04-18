import paddle
arg_1_tensor = paddle.rand([100, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.squeeze(arg_1,arg_2,)
