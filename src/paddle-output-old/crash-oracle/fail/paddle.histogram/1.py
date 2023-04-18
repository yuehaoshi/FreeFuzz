import paddle
arg_1_tensor = paddle.randint(-32768,32768,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 66
arg_3 = "max"
arg_4 = 27
res = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
