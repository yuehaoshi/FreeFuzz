import paddle
arg_1_tensor = paddle.randint(-8,128,[5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.bincount(arg_1,)