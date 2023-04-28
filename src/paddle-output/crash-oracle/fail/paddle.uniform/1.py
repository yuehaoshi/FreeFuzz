import paddle
arg_1_0 = "circular"
arg_1_1 = False
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
arg_3 = 0.0
arg_4 = 0.1
res = paddle.uniform(arg_1,dtype=arg_2,min=arg_3,max=arg_4,)
