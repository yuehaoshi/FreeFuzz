import paddle
arg_1_tensor = paddle.rand([3, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.linalg.pinv(arg_1,)
