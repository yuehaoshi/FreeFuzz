import paddle
arg_1_tensor = paddle.randint(-2,8192,[6, 7, 8], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 0
arg_3 = [arg_3_0,]
arg_4_0 = -105.0
arg_4_1 = -1e+20
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = -2
arg_5_1 = -3
arg_5 = [arg_5_0,arg_5_1,]
res = paddle.strided_slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,strides=arg_5,)
