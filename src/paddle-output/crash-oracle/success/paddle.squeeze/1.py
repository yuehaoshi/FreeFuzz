import paddle
arg_1_tensor = paddle.rand([5, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.squeeze(arg_1,axis=arg_2,)
