import paddle
arg_1_tensor = paddle.randint(-64,128,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.exp(arg_1,)
