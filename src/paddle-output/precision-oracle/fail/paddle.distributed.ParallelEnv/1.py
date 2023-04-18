results = dict()
import paddle
import time
arg_class = paddle.distributed.ParallelEnv()
arg_1_0 = 16
arg_1_1 = -36
arg_1 = (arg_1_0,arg_1_1,)
start = time.time()
results["time_low"] = arg_class(*arg_1)
results["time_low"] = time.time() - start
arg_1 = (arg_1_0,arg_1_1,)
start = time.time()
results["time_high"] = arg_class(*arg_1)
results["time_high"] = time.time() - start

print(results)
