import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 10.0
res = paddle.Tensor.fill_(arg_1,arg_2,)
