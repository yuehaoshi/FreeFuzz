import paddle
arg_1 = "x"
arg_2_0 = 3
arg_2_1 = 100
arg_2_2 = 100
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = "float32"
res = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
