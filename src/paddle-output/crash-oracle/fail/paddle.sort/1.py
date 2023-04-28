import paddle
arg_1_tensor = paddle.rand([1, 30000], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2.0
res = paddle.sort(arg_1,descending=arg_2,)
