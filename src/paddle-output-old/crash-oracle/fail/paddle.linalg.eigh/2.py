import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
arg_2 = -47
res = paddle.linalg.eigh(arg_1,UPLO=arg_2,)
