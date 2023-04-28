import paddle
real = paddle.rand([47, 1024], paddle.float64)
imag = paddle.rand([47, 1024], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 12.0
arg_3 = True
arg_4 = False
res = paddle.signal.stft(arg_1,n_fft=arg_2,center=arg_3,onesided=arg_4,)
