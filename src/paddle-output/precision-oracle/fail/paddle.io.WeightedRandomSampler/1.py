results = dict()
import paddle
import time
arg_1_0 = -1e+20
arg_1_1 = -2.6999999999999993
arg_1_2 = -35.5
arg_1_3 = 1e+20
arg_1_4 = 1.1999999999999993
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,]
arg_2 = -22
arg_3 = False
arg_class = paddle.io.WeightedRandomSampler(weights=arg_1,num_samples=arg_2,replacement=arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
