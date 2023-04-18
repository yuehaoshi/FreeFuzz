import paddle
arg_1 = "builtinsrange"
arg_2_0 = 21
arg_2_1 = 63
arg_2 = [arg_2_0,arg_2_1,]
arg_class = paddle.io.Subset(dataset=arg_1,indices=arg_2,)
arg_3 = None
res = arg_class(*arg_3)
