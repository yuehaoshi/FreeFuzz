results = dict()
import paddle
import time
arg_1_0 = False
arg_1_1 = False
arg_1_2 = "max"
arg_1_3 = False
arg_1_4 = "max"
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,]
arg_2 = 5
arg_3 = 19.0
start = time.time()
results["time_low"] = paddle.io.WeightedRandomSampler(weights=arg_1,num_samples=arg_2,replacement=arg_3,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,]
start = time.time()
results["time_high"] = paddle.io.WeightedRandomSampler(weights=arg_1,num_samples=arg_2,replacement=arg_3,)
results["time_high"] = time.time() - start

print(results)
