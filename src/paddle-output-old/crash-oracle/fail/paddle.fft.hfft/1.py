import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.hfft(arg_1,)
