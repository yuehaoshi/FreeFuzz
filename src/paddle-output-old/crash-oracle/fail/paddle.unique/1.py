import paddle
arg_1_tensor = paddle.randint(-4,8192,[6], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 73
arg_3 = True
arg_4 = True
res = paddle.unique(arg_1,return_index=arg_2,return_inverse=arg_3,return_counts=arg_4,)
