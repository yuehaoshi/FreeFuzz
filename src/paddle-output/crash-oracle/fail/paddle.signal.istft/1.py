import paddle
arg_1_tensor = paddle.rand([8, 257, 376], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = 512
res = paddle.signal.istft(arg_1,n_fft=arg_2,)
