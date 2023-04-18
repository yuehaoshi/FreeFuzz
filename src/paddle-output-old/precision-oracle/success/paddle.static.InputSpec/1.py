results = dict()
import paddle
import time
arg_1_0 = -1
arg_1_1 = 1
arg_1_2 = 28
arg_1_3 = 28
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = "int32"
arg_3 = "image"
start = time.time()
results["time_low"] = paddle.static.InputSpec(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_high"] = paddle.static.InputSpec(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
