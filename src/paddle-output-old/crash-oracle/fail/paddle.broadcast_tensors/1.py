import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.broadcast_tensors(input=arg_1,)
