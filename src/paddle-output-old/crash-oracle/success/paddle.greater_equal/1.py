import paddle
arg_1_tensor = paddle.randint(-64,64,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,32,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.greater_equal(arg_1,arg_2,)
