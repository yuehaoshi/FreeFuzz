import paddle
arg_1_tensor = paddle.randint(-8192,16384,[100, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.zeros_like(arg_1,)
