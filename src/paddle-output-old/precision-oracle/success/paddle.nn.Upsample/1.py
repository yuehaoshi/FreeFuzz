results = dict()
import paddle
import time
arg_1_0 = 25
arg_1_1 = -90
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.nn.Upsample(size=arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.nn.Upsample(size=arg_1,)
results["time_high"] = time.time() - start

print(results)
