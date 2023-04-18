import paddle
arg_1_tensor = paddle.randint(-4,4,[3, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,32768,[], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.crop(arg_1,arg_2,)
