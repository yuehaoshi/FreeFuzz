import paddle
arg_1_tensor = paddle.rand([5, 6, 5, 6, 8], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 1
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "sum"
res = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,)
