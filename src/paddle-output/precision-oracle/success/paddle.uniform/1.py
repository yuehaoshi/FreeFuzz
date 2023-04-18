results = dict()
import paddle
import time
arg_1_0 = 10
arg_1_1 = 10
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = -0.1
arg_3 = 121.1
start = time.time()
results["time_low"] = paddle.uniform(shape=arg_1,min=arg_2,max=arg_3,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.uniform(shape=arg_1,min=arg_2,max=arg_3,)
results["time_high"] = time.time() - start

print(results)
