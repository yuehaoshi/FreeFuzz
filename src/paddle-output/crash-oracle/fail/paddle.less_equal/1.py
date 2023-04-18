import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,64,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.less_equal(arg_1,arg_2,)
