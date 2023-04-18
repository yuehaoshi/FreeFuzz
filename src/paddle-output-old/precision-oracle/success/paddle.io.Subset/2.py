results = dict()
import paddle
import time
arg_1 = 29
arg_2_0 = "sum"
arg_2_1 = "max"
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_low"] = paddle.io.Subset(dataset=arg_1,indices=arg_2,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,]
start = time.time()
results["time_high"] = paddle.io.Subset(dataset=arg_1,indices=arg_2,)
results["time_high"] = time.time() - start

print(results)
