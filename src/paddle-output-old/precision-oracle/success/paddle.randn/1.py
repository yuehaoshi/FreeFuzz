results = dict()
import paddle
import time
arg_1_0 = -42
arg_1_1 = -48
arg_1_2 = 46
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_low"] = paddle.randn(arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_high"] = paddle.randn(arg_1,)
results["time_high"] = time.time() - start

print(results)
