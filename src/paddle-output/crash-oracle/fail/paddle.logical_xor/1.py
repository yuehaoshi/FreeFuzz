import paddle
arg_1_tensor = paddle.randint(-8192,1,[2, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.logical_xor(arg_1,arg_2,)
