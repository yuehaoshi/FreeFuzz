import paddle
arg_1_tensor = paddle.randint(-4,512,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.ones_like(arg_1,)