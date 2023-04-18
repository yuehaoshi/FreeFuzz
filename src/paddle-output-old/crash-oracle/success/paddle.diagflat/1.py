import paddle
arg_1_tensor = paddle.randint(-64,1,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -39
res = paddle.diagflat(arg_1,offset=arg_2,)
