import paddle
arg_1_tensor = paddle.rand([7], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ifft(arg_1,)
