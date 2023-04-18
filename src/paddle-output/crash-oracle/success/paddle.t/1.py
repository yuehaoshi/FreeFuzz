import paddle
arg_1_tensor = paddle.randint(-8192,8192,[2, 3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.t(arg_1,)
