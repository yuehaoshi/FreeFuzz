import paddle
arg_1_0 = False
arg_1_1 = False
arg_1_2 = "max"
arg_1_3 = False
arg_1_4 = "max"
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,]
arg_2 = 5
arg_3 = 19.0
res = paddle.io.WeightedRandomSampler(weights=arg_1,num_samples=arg_2,replacement=arg_3,)
