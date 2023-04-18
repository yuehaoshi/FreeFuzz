import paddle
arg_1_0_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_tensor = paddle.randint(-2,64,[2, 1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.multiplex(arg_1,arg_2,)
