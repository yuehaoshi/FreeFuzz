results = dict()
import paddle
import time
arg_1 = 0.0
arg_2 = 2.0
arg_class = paddle.distribution.Normal(loc=arg_1,scale=arg_2,)
arg_3 = None
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
