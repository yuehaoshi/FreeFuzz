import paddle
arg_1_tensor = paddle.rand([20], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -10
arg_3 = None
res = paddle.roll(arg_1,arg_2,name=arg_3,)
