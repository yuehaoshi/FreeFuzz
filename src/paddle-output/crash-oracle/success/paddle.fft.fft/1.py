import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.fft(arg_1,)