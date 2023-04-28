import paddle
arg_1_0_tensor = paddle.randint(-4,16384,[1], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8,4096,[1], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = paddle.standard_normal(shape=arg_1,)
