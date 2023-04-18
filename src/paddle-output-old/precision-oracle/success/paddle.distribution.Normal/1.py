results = dict()
import paddle
import time
arg_1 = 0.0
arg_2 = 3.0
start = time.time()
results["time_low"] = paddle.distribution.Normal(loc=arg_1,scale=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.distribution.Normal(loc=arg_1,scale=arg_2,)
results["time_high"] = time.time() - start

print(results)
