import paddle
arg_1_tensor = paddle.randint(-16,8192,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -16
res = paddle.argmax(arg_1,axis=arg_2,)
