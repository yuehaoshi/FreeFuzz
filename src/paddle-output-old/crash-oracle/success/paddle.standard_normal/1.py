import paddle
arg_1_tensor = paddle.randint(-16384,1024,[2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.standard_normal(arg_1,)
