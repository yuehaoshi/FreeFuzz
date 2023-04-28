results = dict()
import paddle
import time
arg_1_0 = 2
arg_1_1 = 2
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "paddleVarType"
arg_3 = 0.9
start = time.time()
results["time_low"] = paddle.full(shape=arg_1,dtype=arg_2,fill_value=arg_3,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.full(shape=arg_1,dtype=arg_2,fill_value=arg_3,)
results["time_high"] = time.time() - start

print(results)
