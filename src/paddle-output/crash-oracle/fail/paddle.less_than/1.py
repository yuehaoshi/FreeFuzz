import paddle
arg_1_tensor = paddle.rand([0], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,8,[3], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.less_than(arg_1,arg_2,)
