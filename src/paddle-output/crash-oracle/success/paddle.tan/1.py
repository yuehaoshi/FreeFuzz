import paddle
arg_1_tensor = paddle.rand([2, 4, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.tan(arg_1,)