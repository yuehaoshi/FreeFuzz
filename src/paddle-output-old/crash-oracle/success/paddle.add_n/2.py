import paddle
arg_1_0_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = paddle.add_n(arg_1,)
