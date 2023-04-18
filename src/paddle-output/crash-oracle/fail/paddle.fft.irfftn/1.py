import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -2
arg_3_1 = -1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "circular"
arg_5_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
res = paddle.fft.irfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
