import paddle
arg_1_tensor = paddle.randint(-2,32768,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 29
res = paddle.diagflat(arg_1,offset=arg_2,)
