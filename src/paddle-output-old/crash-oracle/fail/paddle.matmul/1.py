import paddle
arg_1_tensor = paddle.rand([2, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,4096,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = True
res = paddle.matmul(arg_1,arg_2,transpose_y=arg_3,)
