import paddle
arg_1_tensor = paddle.randint(-32768,2048,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.broadcast_to(arg_1,shape=arg_2,)
