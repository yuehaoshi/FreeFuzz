import paddle
arg_1_tensor = paddle.rand([1, 1, 3, 4, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 11
arg_3 = False
arg_4 = -1007
res = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
