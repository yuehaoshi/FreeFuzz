import paddle
arg_1_tensor = paddle.rand([3, 4, 5, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 2
arg_2_2 = 3
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3_0 = -3
arg_3_1 = 0
arg_3_2 = 2
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
arg_4_0 = 3
arg_4_1 = 2
arg_4_2 = 4
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5_0 = 1
arg_5_1 = 1
arg_5_2 = 1
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
res = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
