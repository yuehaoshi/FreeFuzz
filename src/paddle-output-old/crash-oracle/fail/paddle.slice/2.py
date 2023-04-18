import paddle
arg_1_tensor = paddle.randint(-8192,16384,[7], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3_0 = 5
arg_3 = [arg_3_0,]
arg_4_0 = 6
arg_4 = [arg_4_0,]
res = paddle.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
