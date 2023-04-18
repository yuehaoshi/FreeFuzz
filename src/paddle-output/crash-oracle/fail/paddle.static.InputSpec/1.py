import paddle
arg_1_0 = -1
arg_1_1 = 784
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "circular"
arg_3 = "x"
arg_class = paddle.static.InputSpec(arg_1,arg_2,arg_3,)
arg_4 = None
res = arg_class(*arg_4)
