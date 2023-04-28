import paddle
arg_1_tensor = paddle.randint(-256,16384,[64, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 31
arg_2_1 = -36
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.reshape(arg_1,arg_2,)
