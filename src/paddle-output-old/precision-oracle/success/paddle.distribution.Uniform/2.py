results = dict()
import paddle
import time
arg_1 = 3.0
arg_2 = 16
start = time.time()
results["time_low"] = paddle.distribution.Uniform(low=arg_1,high=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.distribution.Uniform(low=arg_1,high=arg_2,)
results["time_high"] = time.time() - start

print(results)
