import paddle
arg_1_tensor = paddle.rand([3, 2, 1, 3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 6
arg_2_1 = 6
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
res = paddle.fft.ifftn(arg_1,arg_2,arg_3,arg_4,)
