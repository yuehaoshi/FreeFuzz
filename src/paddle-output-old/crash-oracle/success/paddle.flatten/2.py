import paddle
arg_1_tensor = paddle.randint(-8192,1024,[1, 2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3 = 2
res = paddle.flatten(arg_1,start_axis=arg_2,stop_axis=arg_3,)
