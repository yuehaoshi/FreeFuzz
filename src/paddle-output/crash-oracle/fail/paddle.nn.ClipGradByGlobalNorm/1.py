import paddle
arg_1 = 1.0
arg_class = paddle.nn.ClipGradByGlobalNorm(clip_norm=arg_1,)
arg_2 = None
res = arg_class(*arg_2)
