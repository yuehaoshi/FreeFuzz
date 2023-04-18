import paddle
arg_1_tensor = paddle.randint(-32768,256,[3, 9, 5], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
arg_3 = 1
res = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)
