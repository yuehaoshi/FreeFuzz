import paddle
arg_1_tensor = paddle.randint(-16,8,[5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.zero_(arg_1,)
