import paddle
arg_1_tensor = paddle.randint(-8192,16,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,16384,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.less_equal(arg_1,arg_2,)