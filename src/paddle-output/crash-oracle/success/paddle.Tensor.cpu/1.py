import paddle
arg_1_tensor = paddle.randint(-8,8192,[1, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.cpu(arg_1,)
