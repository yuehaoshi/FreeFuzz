import paddle
arg_1 = "max"
arg_2 = 0.02
arg_3_0 = 8
arg_3_1 = 8
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
