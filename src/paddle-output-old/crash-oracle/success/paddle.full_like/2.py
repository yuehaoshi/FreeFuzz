import paddle
arg_1_tensor = paddle.rand([2, 3, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 44.0
res = paddle.full_like(arg_1,arg_2,)
