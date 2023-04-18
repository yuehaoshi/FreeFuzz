import paddle
arg_1_tensor = paddle.randint(-32768,32,[4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,128,[4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.floor_divide(arg_1,arg_2,)
