import paddle
arg_1_tensor = paddle.randint(-256,2048,[4, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 17
arg_2_1 = -28
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.tile(arg_1,arg_2,)
