results = dict()
import paddle
import time
arg_1_0 = 1
arg_1 = [arg_1_0,]
arg_2 = 10
arg_3 = "float32"
start = time.time()
results["time_low"] = paddle.full(shape=arg_1,fill_value=arg_2,dtype=arg_3,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,]
start = time.time()
results["time_high"] = paddle.full(shape=arg_1,fill_value=arg_2,dtype=arg_3,)
results["time_high"] = time.time() - start

print(results)
