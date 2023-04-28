import paddle
arg_1 = 79.0
arg_2_0_0 = 1
arg_2_0_1 = 1
arg_2_0_2 = 28
arg_2_0_3 = 28
arg_2_0 = [arg_2_0_0,arg_2_0_1,arg_2_0_2,arg_2_0_3,]
arg_2_1_0 = 1
arg_2_1_1 = 400
arg_2_1 = [arg_2_1_0,arg_2_1_1,]
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = "float32"
arg_3_1 = "float32"
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.summary(arg_1,arg_2,dtypes=arg_3,)
