import paddle
arg_1_tensor = paddle.rand([2, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 38
res = paddle.logsumexp(arg_1,axis=arg_2,)
