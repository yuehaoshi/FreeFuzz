import paddle
arg_1_tensor = paddle.randint(-128,8192,[1, 2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,32,[], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.subtract(arg_1,arg_2,)
