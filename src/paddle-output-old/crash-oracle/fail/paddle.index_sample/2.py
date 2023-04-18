import paddle
arg_1_tensor = paddle.randint(-2048,32768,[3, 4], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,16384,[3, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.index_sample(arg_1,arg_2,)
