results = dict()
import paddle
import time
real = paddle.rand([47, 1024], paddle.float32)
imag = paddle.rand([47, 1024], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 12.0
arg_3 = True
arg_4 = False
start = time.time()
results["time_low"] = paddle.signal.stft(arg_1,n_fft=arg_2,center=arg_3,onesided=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.complex128)
start = time.time()
results["time_high"] = paddle.signal.stft(arg_1,n_fft=arg_2,center=arg_3,onesided=arg_4,)
results["time_high"] = time.time() - start

print(results)
