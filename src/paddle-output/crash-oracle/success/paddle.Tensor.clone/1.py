import paddle
arg_1_tensor = paddle.rand([8, 7], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.clone(arg_1,)