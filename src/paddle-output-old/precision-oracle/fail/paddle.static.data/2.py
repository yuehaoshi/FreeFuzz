results = dict()
import paddle
import time
arg_1 = "x"
arg_2_0 = 3
arg_2_1 = 2
arg_2_2 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = paddle.static.data(name=arg_1,shape=arg_2,)
results["time_low"] = time.time() - start
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.static.data(name=arg_1,shape=arg_2,)
results["time_high"] = time.time() - start

print(results)
