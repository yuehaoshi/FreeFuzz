results = dict()
import paddle
import time
arg_1_0 = 2
arg_1_1 = 1
arg_1_2 = 3
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2_0 = 1
arg_2_1 = 3
arg_2_2 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = paddle.broadcast_shape(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.broadcast_shape(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
