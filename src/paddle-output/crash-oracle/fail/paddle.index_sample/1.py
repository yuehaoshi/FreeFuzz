import paddle
arg_1_tensor = paddle.randint(-4,512,[3, 4], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,1,[3, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.index_sample(arg_1,arg_2,)
