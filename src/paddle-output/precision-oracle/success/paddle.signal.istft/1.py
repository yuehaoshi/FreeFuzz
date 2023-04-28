results = dict()
import paddle
import time
real = paddle.rand([8, 257, 376], paddle.float32)
imag = paddle.rand([8, 257, 376], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 512
start = time.time()
results["time_low"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.complex128)
start = time.time()
results["time_high"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
results["time_high"] = time.time() - start

print(results)
