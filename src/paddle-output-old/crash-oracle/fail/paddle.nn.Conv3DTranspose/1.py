import paddle
arg_1 = 4
arg_2 = 6
arg_3_0 = -56
arg_3_1 = 77
arg_3_2 = -29
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
res = paddle.nn.Conv3DTranspose(arg_1,arg_2,arg_3,)
