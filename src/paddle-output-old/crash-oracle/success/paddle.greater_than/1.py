import paddle
arg_1_tensor = paddle.randint(-16384,4,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,2,[3, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.greater_than(arg_1,arg_2,)
