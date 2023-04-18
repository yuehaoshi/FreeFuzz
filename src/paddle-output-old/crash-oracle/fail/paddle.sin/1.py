import paddle
arg_1_tensor = paddle.randint(-8192,8,[4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.sin(arg_1,)
