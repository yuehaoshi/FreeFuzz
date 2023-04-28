import paddle
arg_1 = "X"
arg_2_0 = "max"
arg_2_1 = False
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "float"
res = paddle.static.data(name=arg_1,shape=arg_2,dtype=arg_3,)
