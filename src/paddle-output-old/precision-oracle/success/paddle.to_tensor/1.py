results = dict()
import paddle
import time
arg_1_0 = -16
arg_1_1 = 23
arg_1_2 = 45
arg_1_3 = 53
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_low"] = paddle.to_tensor(arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_high"] = paddle.to_tensor(arg_1,)
results["time_high"] = time.time() - start

print(results)
