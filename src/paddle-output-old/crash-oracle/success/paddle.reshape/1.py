import paddle
arg_1_tensor = paddle.randint(-1024,4096,[12], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 4
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.reshape(arg_1,arg_2,)
