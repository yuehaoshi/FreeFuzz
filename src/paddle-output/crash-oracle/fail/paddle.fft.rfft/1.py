import paddle
arg_1_tensor = paddle.rand([6, 5, 5, 7, 7], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 58
arg_3 = "max"
arg_4 = "backward"
res = paddle.fft.rfft(arg_1,arg_2,arg_3,arg_4,)
