import paddle
arg_1_tensor = paddle.rand([2, 2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 5.0
arg_2_1 = True
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.prod(arg_1,arg_2,)
