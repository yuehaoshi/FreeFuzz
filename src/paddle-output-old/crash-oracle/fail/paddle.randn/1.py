import paddle
arg_1_0 = 17
arg_1_1 = 1024
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "paddleVarType"
res = paddle.randn(arg_1,dtype=arg_2,)
