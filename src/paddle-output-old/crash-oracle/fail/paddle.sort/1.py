import paddle
arg_1_tensor = paddle.rand([2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1024
res = paddle.sort(x=arg_1,axis=arg_2,)
