import paddle
arg_1_tensor = paddle.randint(-2,128,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 100
arg_2 = [arg_2_0,]
res = paddle.expand(arg_1,shape=arg_2,)
