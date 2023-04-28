import paddle
arg_1_tensor = paddle.randint(-1,16384,[3, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 9, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 1024
arg_3_1 = 58
arg_3_2 = -21
arg_3_3 = 16
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
res = paddle.scatter_nd(arg_1,arg_2,arg_3,)
