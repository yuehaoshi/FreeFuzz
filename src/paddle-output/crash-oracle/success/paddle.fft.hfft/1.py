import paddle
real = paddle.rand([4, 4, 4], paddle.float64)
imag = paddle.rand([4, 4, 4], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = -1
arg_4 = "backward"
res = paddle.fft.hfft(arg_1,arg_2,arg_3,arg_4,)
