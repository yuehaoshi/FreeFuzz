import paddle
arg_1_tensor = paddle.rand([3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,4096,[44, 3, 1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.index_sample(arg_1,arg_2,)
