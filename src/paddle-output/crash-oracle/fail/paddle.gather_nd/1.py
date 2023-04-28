import paddle
arg_1_tensor = paddle.rand([3, 5, 9, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,8192,[1, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.gather_nd(arg_1,arg_2,)
