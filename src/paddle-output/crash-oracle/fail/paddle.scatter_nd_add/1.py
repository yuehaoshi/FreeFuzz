import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,8192,[3, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16384,2,[30000], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
res = paddle.scatter_nd_add(arg_1,arg_2,arg_3,)
