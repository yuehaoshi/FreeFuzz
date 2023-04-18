import paddle
arg_1_0 = 2
arg_1_1 = 3
arg_1_2 = 4
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = "int32"
res = paddle.rand(arg_1,dtype=arg_2,)
