import paddle
arg_1_tensor = paddle.randint(-64,4096,[0, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,16384,[3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.incubate.segment_min(arg_1,arg_2,)
