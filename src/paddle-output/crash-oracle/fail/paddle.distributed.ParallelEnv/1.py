import paddle
arg_class = paddle.distributed.ParallelEnv()
arg_1_0 = 16
arg_1_1 = -36
arg_1 = (arg_1_0,arg_1_1,)
res = arg_class(*arg_1)
