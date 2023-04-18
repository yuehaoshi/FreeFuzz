import paddle
arg_1 = -1
arg_2 = "zeros"
arg_class = paddle.nn.ClipGradByValue(min=arg_1,max=arg_2,)
arg_3 = None
res = arg_class(*arg_3)
