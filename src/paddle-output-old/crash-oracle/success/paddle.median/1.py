import paddle
arg_1_tensor = paddle.randint(-2048,32,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = True
res = paddle.median(arg_1,axis=arg_2,keepdim=arg_3,)
