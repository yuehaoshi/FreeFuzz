import paddle
arg_1_tensor = paddle.rand([3, 4, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([4, 3, 2], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3_0_0 = 1
arg_3_0_1 = 0
arg_3_0 = [arg_3_0_0,arg_3_0_1,]
arg_3_1_0 = 0
arg_3_1_1 = 1
arg_3_1 = [arg_3_1_0,arg_3_1_1,]
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
