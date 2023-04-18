import paddle
arg_1_tensor = paddle.randint(-32768,16,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.hfft(arg_1,)
