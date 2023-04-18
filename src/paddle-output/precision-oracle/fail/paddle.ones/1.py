results = dict()
import paddle
import time
arg_1_0 = 4
arg_1_1 = 60
arg_1_2 = -57
arg_1_3 = -11
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_low"] = paddle.ones(shape=arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_high"] = paddle.ones(shape=arg_1,)
results["time_high"] = time.time() - start

print(results)
