import paddle
arg_1_tensor = paddle.rand([10, 1, 5, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1, 3, 2, 5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.matmul(arg_1,arg_2,)
