import paddle
arg_1_tensor = paddle.randint(-8192,2048,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.min(arg_1,)
