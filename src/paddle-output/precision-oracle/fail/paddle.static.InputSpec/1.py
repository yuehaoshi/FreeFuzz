results = dict()
import paddle
import time
arg_1_0 = -1
arg_1_1 = 784
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "circular"
arg_3 = "x"
arg_class = paddle.static.InputSpec(arg_1,arg_2,arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
