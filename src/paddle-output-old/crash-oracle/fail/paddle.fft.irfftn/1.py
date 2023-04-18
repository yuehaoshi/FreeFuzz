import paddle
arg_1_tensor = paddle.randint(-8,2048,[3], dtype=paddle.uint16)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.irfftn(arg_1,)
