results = dict()
import paddle
import time
arg_1_0 = -51
arg_1_1 = 28
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
start = time.time()
results["time_low"] = paddle.zeros(shape=arg_1,dtype=arg_2,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.zeros(shape=arg_1,dtype=arg_2,)
results["time_high"] = time.time() - start

print(results)
