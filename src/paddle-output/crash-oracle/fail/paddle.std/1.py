import paddle
real = paddle.rand([2, 0], paddle.float32)
imag = paddle.rand([2, 0], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
res = paddle.std(arg_1,)
