import paddle
arg_1_tensor = paddle.randint(-64,16,[2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.randn(arg_1,)
