import paddle
arg_1_0_tensor = paddle.rand([10, 2, 5], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_0_tensor = paddle.rand([10, 2, 5], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3 = None
arg_4 = None
res = paddle.static.gradients(arg_1,arg_2,arg_3,arg_4,)
