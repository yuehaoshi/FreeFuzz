import paddle
real = paddle.rand([2, 0], paddle.float32)
imag = paddle.rand([2, 0], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = -9
res = paddle.moveaxis(arg_1,arg_2,arg_3,)
