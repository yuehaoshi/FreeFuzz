import paddle
arg_1_0_tensor = paddle.rand([-1, 608, 14, 14], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([-1, 32, 14, 14], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 1
res = paddle.concat(arg_1,axis=arg_2,)
