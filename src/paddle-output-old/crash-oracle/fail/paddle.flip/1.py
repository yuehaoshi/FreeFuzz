import paddle
arg_1_tensor = paddle.rand([3, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 52
arg_2_1 = -15
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.flip(arg_1,arg_2,)
