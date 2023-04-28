import paddle
arg_1_0 = 4
arg_1_1 = 5
arg_1_2 = 6
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = "float32"
res = paddle.rand(shape=arg_1,dtype=arg_2,)
