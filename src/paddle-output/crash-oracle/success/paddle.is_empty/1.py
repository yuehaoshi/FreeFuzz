import paddle
arg_1_tensor = paddle.rand([4, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.is_empty(x=arg_1,)
