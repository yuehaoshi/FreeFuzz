import paddle
arg_1_tensor = paddle.randint(-2,1024,[2, 3, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,128,[1, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.gather_nd(arg_1,arg_2,)
