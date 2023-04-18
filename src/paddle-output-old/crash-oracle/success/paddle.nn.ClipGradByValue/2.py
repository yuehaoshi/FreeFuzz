import paddle
arg_1 = 40
arg_2 = 52
res = paddle.nn.ClipGradByValue(min=arg_1,max=arg_2,)
