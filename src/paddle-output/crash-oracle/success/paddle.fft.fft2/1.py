import paddle
real = paddle.rand([8, 2, 7, 6, 9], paddle.float64)
imag = paddle.rand([8, 2, 7, 6, 9], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
res = paddle.fft.fft2(arg_1,arg_2,arg_3,arg_4,)
