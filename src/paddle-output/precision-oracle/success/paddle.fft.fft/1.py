results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = -1
arg_4 = "backward"
start = time.time()
results["time_low"] = paddle.fft.fft(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.fft.fft(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
