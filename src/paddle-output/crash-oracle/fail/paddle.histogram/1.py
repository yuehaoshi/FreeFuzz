import paddle
arg_1_tensor = paddle.randint(-4096,16,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -30
arg_3 = -50
arg_4 = 3
res = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
