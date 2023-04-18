import paddle
arg_1_tensor = paddle.randint(-32,8,[3, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 9, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 2
arg_3_1 = 3
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.scatter_nd(arg_1,arg_2,arg_3,)
