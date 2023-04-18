import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -37
arg_3_1 = 29
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = "ignore_nan"
res = paddle.fft.hfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
