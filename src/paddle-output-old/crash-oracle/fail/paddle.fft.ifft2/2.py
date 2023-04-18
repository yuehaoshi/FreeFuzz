import paddle
arg_1_tensor = paddle.rand([5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ifft2(arg_1,)
