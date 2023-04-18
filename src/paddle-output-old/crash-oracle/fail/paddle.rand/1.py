import paddle
arg_1_0 = 31
arg_1_1 = 4
arg_1_2 = -16
arg_1_3 = 66
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = "paddleVarType"
res = paddle.rand(arg_1,dtype=arg_2,)
