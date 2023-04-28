results = dict()
import paddle
import time
arg_1 = -3.141592653589793
arg_2 = True
arg_3 = 469
arg_4 = "int64"
start = time.time()
results["time_low"] = paddle.linspace(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.linspace(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_high"] = time.time() - start

print(results)
