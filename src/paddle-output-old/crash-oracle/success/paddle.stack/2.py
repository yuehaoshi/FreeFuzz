import paddle
arg_1_0_tensor = paddle.rand([1, 2], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([1, 2], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.rand([1, 2], dtype=paddle.float32)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = 0
res = paddle.stack(arg_1,axis=arg_2,)
