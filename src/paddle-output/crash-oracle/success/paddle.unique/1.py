import paddle
arg_1_tensor = paddle.randint(-2,8192,[3, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.unique(arg_1,)
