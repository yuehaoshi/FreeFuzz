import paddle
arg_1_0 = 2
arg_1_1 = 8
arg_1_2 = 32
arg_1_3 = 32
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = 1
arg_3 = 30
res = paddle.nn.SpectralNorm(arg_1,dim=arg_2,power_iters=arg_3,)
