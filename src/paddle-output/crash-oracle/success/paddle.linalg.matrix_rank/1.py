import paddle
arg_1_tensor = paddle.rand([1, 7, 9, 3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.01
arg_3 = False
res = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
