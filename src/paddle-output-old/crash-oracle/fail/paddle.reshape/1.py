import paddle
arg_1_tensor = paddle.rand([2, 4, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.randint(-16384,8192,[1], dtype=paddle.int32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = paddle.reshape(arg_1,shape=arg_2,)
