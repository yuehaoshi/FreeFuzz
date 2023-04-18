import paddle
arg_1_0_tensor = paddle.randint(-8192,1024,[2, 3], dtype=paddle.int64)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-256,4,[2, 3], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.randint(-16384,256,[2, 2], dtype=paddle.int64)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = 1
res = paddle.concat(x=arg_1,axis=arg_2,)
