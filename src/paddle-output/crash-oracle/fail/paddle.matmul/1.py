import paddle
arg_1_tensor = paddle.rand([4, 2, 1, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048,4096,[24, 2, 11], dtype=paddle.bfloat16)
arg_2 = arg_2_tensor.clone()
arg_3 = True
res = paddle.matmul(x=arg_1,y=arg_2,transpose_y=arg_3,)
