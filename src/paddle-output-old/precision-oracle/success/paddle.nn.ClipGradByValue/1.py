results = dict()
import paddle
import time
arg_1 = -1
arg_2 = -19
start = time.time()
results["time_low"] = paddle.nn.ClipGradByValue(min=arg_1,max=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.ClipGradByValue(min=arg_1,max=arg_2,)
results["time_high"] = time.time() - start

print(results)
