import paddle
arg_1_tensor = paddle.rand([513], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.cos(arg_1,)
