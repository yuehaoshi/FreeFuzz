import paddle
arg_1_tensor = paddle.rand([1, 3, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ifft(arg_1,)
