results = dict()
import paddle
import time
arg_1 = 0
arg_2 = 35
arg_3 = 64
arg_4 = "float32"
start = time.time()
results["time_low"] = paddle.linspace(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.linspace(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
