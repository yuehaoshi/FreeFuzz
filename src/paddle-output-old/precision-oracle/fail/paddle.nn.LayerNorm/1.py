results = dict()
import paddle
import time
arg_1_0 = -1
arg_1_1 = -49
arg_1_2 = -11
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_low"] = paddle.nn.LayerNorm(arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_high"] = paddle.nn.LayerNorm(arg_1,)
results["time_high"] = time.time() - start

print(results)
