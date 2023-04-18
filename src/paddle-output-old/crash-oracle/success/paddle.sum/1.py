import paddle
arg_1_tensor = paddle.randint(-512,4,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.sum(arg_1,axis=arg_2,)
