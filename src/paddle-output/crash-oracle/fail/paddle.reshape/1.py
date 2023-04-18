import paddle
arg_1_tensor = paddle.rand([2, 4, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 5
arg_2_2 = 9
arg_2_3 = 10
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
res = paddle.reshape(arg_1,shape=arg_2,)
