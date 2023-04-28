import paddle
arg_1_tensor = paddle.randint(-2,16384,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,2,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.bitwise_and(arg_1,arg_2,)
