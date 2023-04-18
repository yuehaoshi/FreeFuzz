import paddle
arg_1 = 128
arg_2 = 2
arg_3 = 29
arg_4 = 64
arg_5 = 512
arg_class = paddle.nn.Transformer(arg_1,arg_2,arg_3,arg_4,arg_5,)
arg_6 = "max"
res = arg_class(*arg_6)
