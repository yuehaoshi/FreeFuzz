import paddle
arg_1 = 29
arg_2_0 = "sum"
arg_2_1 = "max"
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.io.Subset(dataset=arg_1,indices=arg_2,)
