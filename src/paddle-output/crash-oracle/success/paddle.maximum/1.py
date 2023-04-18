import paddle
arg_1_tensor = paddle.randint(-32,128,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,16,[2, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.maximum(arg_1,arg_2,)
