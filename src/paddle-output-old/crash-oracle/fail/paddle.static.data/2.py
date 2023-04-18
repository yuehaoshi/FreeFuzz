import paddle
arg_1 = "x"
arg_2_0 = 3
arg_2_1 = 2
arg_2_2 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
res = paddle.static.data(name=arg_1,shape=arg_2,)
