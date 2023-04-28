results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([1, 1, 3, 4, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 11
arg_3 = False
arg_4 = -1007
start = time.time()
results["time_low"] = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
start = time.time()
results["time_high"] = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
