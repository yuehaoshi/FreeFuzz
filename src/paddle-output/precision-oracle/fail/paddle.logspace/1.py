results = dict()
import paddle
import time
arg_1 = 0
arg_2 = 10
arg_3 = 5
arg_4 = "max"
arg_5 = "float32"
start = time.time()
results["time_low"] = paddle.logspace(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.logspace(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)
