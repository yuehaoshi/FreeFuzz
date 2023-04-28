import paddle
arg_1_tensor = paddle.rand([10], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2 = [arg_2_0,]
res = paddle.fft.fftshift(arg_1,arg_2,)
