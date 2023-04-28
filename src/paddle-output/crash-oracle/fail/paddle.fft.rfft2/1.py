import paddle
arg_1_tensor = paddle.rand([8, 8, 6, 4, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 51
arg_3_1 = 3
arg_3 = (arg_3_0,arg_3_1,)
arg_4 = "backward"
res = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
