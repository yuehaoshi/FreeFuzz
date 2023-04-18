import paddle
arg_1_tensor = paddle.randint(-4096,8192,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 4.0
res = paddle.pow(arg_1,arg_2,)
