results = dict()
import paddle
import time
arg_1_0 = -56.5
arg_1 = [arg_1_0,]
arg_2_0 = "max"
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = paddle.distribution.Normal(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,]
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.distribution.Normal(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)