results = dict()
import paddle
import time
arg_1 = 1017
arg_2 = 43.5
start = time.time()
results["time_low"] = paddle.fft.fftfreq(arg_1,d=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.fft.fftfreq(arg_1,d=arg_2,)
results["time_high"] = time.time() - start

print(results)
