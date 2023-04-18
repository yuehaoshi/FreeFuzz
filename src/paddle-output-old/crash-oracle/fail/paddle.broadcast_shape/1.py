import paddle
arg_1_0 = 2
arg_1_1 = 1
arg_1_2 = 3
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2_0 = 75
arg_2_1 = -54
arg_2_2 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = paddle.broadcast_shape(arg_1,arg_2,)
