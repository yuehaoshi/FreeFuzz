results = dict()
import paddle
import time
arg_1_0 = 1
arg_1 = [arg_1_0,]
arg_2 = 4
arg_3 = "int32"
start = time.time()
results["time_low"] = paddle.full(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = paddle.full(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
