import paddle
arg_1_tensor = paddle.rand([1, 3, 64], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.shape(arg_1,)