results = dict()
import paddle
import time
arg_1_0 = 12
arg_1_1 = 12
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.nn.LocalResponseNorm(size=arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.nn.LocalResponseNorm(size=arg_1,)
results["time_high"] = time.time() - start

print(results)
