import paddle
arg_1_tensor = paddle.randint(-1024,1024,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 3, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.bitwise_and(arg_1,arg_2,)
