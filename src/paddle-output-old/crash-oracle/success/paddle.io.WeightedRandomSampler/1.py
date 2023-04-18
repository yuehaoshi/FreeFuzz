import paddle
arg_1_0 = -1e+20
arg_1_1 = -19.7
arg_1_2 = -22.5
arg_1_3 = 1e+20
arg_1_4 = -30.8
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,]
arg_2 = 5
arg_3 = False
res = paddle.io.WeightedRandomSampler(weights=arg_1,num_samples=arg_2,replacement=arg_3,)
