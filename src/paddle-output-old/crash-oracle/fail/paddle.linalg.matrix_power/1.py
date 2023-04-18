import paddle
arg_1_tensor = paddle.randint(-16,32768,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -2
res = paddle.linalg.matrix_power(arg_1,arg_2,)
