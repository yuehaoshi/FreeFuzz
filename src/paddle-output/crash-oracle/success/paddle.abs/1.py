import paddle
arg_1_tensor = paddle.randint(-32768,512,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.abs(arg_1,)
