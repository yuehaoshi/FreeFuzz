import paddle
arg_1_tensor = paddle.randint(-8,16384,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[2, 2], dtype=paddle.bool)
arg_2 = arg_2_tensor.clone()
res = paddle.equal_all(arg_1,arg_2,)