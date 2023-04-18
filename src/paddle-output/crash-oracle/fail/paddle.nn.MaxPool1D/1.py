import paddle
arg_1 = 3
arg_2 = -1
arg_3 = "replicate"
arg_class = paddle.nn.MaxPool1D(kernel_size=arg_1,stride=arg_2,padding=arg_3,)
arg_4 = None
res = arg_class(*arg_4)
