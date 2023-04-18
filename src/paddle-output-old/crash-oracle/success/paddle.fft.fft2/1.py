import paddle
arg_1_tensor = paddle.randint(-8,2048,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.fft2(arg_1,)
