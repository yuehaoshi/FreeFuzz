import paddle
arg_1 = 3
arg_2 = 2
arg_3 = "max"
arg_class = paddle.nn.AvgPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
arg_4 = None
res = arg_class(*arg_4)
