import paddle
arg_class = paddle.nn.LeakyReLU()
arg_1_0 = -1024
arg_1_1 = 20
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
