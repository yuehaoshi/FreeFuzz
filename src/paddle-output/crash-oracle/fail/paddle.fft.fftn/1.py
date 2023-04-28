import paddle
real = paddle.rand([9, 8, 7, 5, 9], paddle.float64)
imag = paddle.rand([9, 8, 7, 5, 9], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 4
arg_2_1 = 4
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 28
arg_3_1 = -12
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = None
res = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
