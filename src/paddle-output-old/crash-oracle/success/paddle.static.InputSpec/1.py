import paddle
arg_1_0 = -1
arg_1_1 = 1
arg_1_2 = 28
arg_1_3 = 28
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = "int32"
arg_3 = "image"
res = paddle.static.InputSpec(arg_1,arg_2,arg_3,)
