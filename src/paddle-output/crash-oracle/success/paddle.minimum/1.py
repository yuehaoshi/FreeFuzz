import paddle
arg_1_tensor = paddle.randint(-32768,8,[1, 2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[3])
arg_2 = arg_2_tensor.clone()
res = paddle.minimum(arg_1,arg_2,)
