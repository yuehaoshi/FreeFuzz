import paddle
arg_1_0 = -1024
arg_1_1 = 1024
arg_1_2 = -1
arg_1_3 = -17
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = -12
arg_3 = 30
arg_class = paddle.nn.SpectralNorm(arg_1,dim=arg_2,power_iters=arg_3,)
arg_4 = None
res = arg_class(*arg_4)
