import paddle
arg_1_tensor = paddle.randint(-16384,1,[6, 2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 0
arg_2_2 = 2
arg_2_3 = -1
arg_2_4 = 3
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,]
res = paddle.sparse.reshape(arg_1,arg_2,)
