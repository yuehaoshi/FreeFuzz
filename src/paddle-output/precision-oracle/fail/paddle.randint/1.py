results = dict()
import paddle
import time
arg_1 = 1024
arg_2 = 32
arg_3_0 = 62
arg_3 = [arg_3_0,]
arg_4 = -51.0
start = time.time()
results["time_low"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,]
start = time.time()
results["time_high"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_high"] = time.time() - start

print(results)
