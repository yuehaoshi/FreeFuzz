import paddle
arg_1_tensor = paddle.randint(-16,1,[4, 0, 4], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1018
arg_3 = 1
arg_4 = "ortho"
res = paddle.fft.irfft2(arg_1,arg_2,arg_3,arg_4,)
