import paddle
arg_1_tensor = paddle.rand([4, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 3], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
res = paddle.multiply(arg_1,arg_2,)
