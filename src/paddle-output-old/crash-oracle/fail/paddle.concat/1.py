import paddle
arg_1_0_tensor = paddle.randint(0,64,[2, 0, 0], dtype=paddle.uint8)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-2,32768,[2, 3], dtype=paddle.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 0
res = paddle.concat(x=arg_1,axis=arg_2,)
