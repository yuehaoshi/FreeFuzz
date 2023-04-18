import paddle
arg_1_tensor = paddle.rand([2, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -11
res = paddle.prod(arg_1,arg_2,)
