results = dict()
import paddle
import time
arg_1 = 0
arg_2 = 10
arg_3 = 87
arg_4 = False
start = time.time()
results["time_low"] = paddle.linspace(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.linspace(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
