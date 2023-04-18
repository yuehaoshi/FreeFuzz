import paddle
arg_1_0_tensor = paddle.randint(-16,32768,[2, 3], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-512,128,[2, 3], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 0
res = paddle.concat(x=arg_1,axis=arg_2,)
