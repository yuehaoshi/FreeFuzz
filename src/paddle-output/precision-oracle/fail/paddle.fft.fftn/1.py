results = dict()
import paddle
import time
real = paddle.rand([9, 8, 7, 5, 9], paddle.float32)
imag = paddle.rand([9, 8, 7, 5, 9], paddle.float32)
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
start = time.time()
results["time_low"] = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.complex128)
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)
