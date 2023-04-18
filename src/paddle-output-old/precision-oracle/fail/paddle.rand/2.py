results = dict()
import paddle
import time
arg_1_0 = 2
arg_1_1 = 3
arg_1_2 = 4
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = "int32"
start = time.time()
results["time_low"] = paddle.rand(arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_high"] = paddle.rand(arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)
