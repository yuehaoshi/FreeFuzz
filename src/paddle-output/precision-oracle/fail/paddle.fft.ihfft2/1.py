results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([1, 1, 3, 1, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 11
arg_2_1 = 11
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = False
start = time.time()
results["time_low"] = paddle.fft.ihfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.fft.ihfft2(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
