import paddle
arg_1_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.std(arg_1,)
