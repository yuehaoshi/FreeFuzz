import paddle
arg_1_0_tensor = paddle.rand([-1, 4, 8, 8], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_tensor = paddle.rand([-1, 2, 8, 8], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.static.gradients(arg_1,arg_2,)
