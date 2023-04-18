results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0.0
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = paddle.fft.ifftn(arg_1,axes=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.fft.ifftn(arg_1,axes=arg_2,)
results["time_high"] = time.time() - start

print(results)
