import paddle
arg_1_tensor = paddle.randint(-1,1,[2, 6], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,256,[2, 4], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.searchsorted(arg_1,arg_2,)