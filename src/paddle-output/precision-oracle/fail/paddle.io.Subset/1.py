results = dict()
import paddle
import time
arg_1 = "builtinsrange"
arg_2_0 = 21
arg_2_1 = 63
arg_2 = [arg_2_0,arg_2_1,]
arg_class = paddle.io.Subset(dataset=arg_1,indices=arg_2,)
arg_3 = None
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
