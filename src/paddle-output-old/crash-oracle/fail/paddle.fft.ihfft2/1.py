import paddle
arg_1_tensor = paddle.randint(-1,64,[5, 5], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
res = paddle.fft.ihfft2(arg_1,)
