import paddle
arg_1_tensor = paddle.randint(-16,2048,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -2
arg_3_1 = -1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = None
res = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
