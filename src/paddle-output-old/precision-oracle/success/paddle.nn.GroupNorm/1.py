results = dict()
import paddle
import time
arg_1 = 6
arg_2 = -47
start = time.time()
results["time_low"] = paddle.nn.GroupNorm(num_channels=arg_1,num_groups=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.GroupNorm(num_channels=arg_1,num_groups=arg_2,)
results["time_high"] = time.time() - start

print(results)
