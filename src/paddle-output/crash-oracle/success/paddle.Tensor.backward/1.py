import paddle
arg_1_tensor = paddle.randint(-16,4,[1, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
res = paddle.Tensor.backward(arg_1,)
