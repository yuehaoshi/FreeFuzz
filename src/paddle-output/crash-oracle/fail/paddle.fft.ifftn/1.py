import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0.0
arg_2 = [arg_2_0,]
res = paddle.fft.ifftn(arg_1,axes=arg_2,)