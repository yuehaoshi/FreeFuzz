import paddle
arg_1_tensor = paddle.rand([2, 9, 6, 5, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "ortho"
arg_5 = None
res = paddle.fft.rfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
