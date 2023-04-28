import paddle
arg_1_tensor = paddle.randint(-1,4,[1, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -1.0
arg_2_1 = True
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.transpose(arg_1,arg_2,)
