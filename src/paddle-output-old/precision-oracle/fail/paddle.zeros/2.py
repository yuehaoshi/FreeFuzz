results = dict()
import paddle
import time
arg_1_0 = -29
arg_1_1 = 5
arg_1_2 = -64
arg_1_3 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = "float32"
start = time.time()
results["time_low"] = paddle.zeros(shape=arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_high"] = paddle.zeros(shape=arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)
