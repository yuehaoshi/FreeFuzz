import paddle
arg_1_tensor = paddle.rand([2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = -1
arg_4 = "backward"
res = paddle.fft.fft(arg_1,arg_2,arg_3,arg_4,)
