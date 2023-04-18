import paddle
arg_1_tensor = paddle.randint(-256,16384,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,64,[2, 3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.expand_as(arg_1,arg_2,)
