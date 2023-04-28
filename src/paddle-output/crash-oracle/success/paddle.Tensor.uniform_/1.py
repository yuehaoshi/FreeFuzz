import paddle
arg_1_tensor = paddle.rand([2, 9, 6, 5, 1], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.uniform_(arg_1,)
