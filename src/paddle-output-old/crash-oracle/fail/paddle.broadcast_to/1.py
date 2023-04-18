import paddle
arg_1_tensor = paddle.randint(-2048,128,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 53
arg_2_1 = 56
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.broadcast_to(arg_1,shape=arg_2,)
