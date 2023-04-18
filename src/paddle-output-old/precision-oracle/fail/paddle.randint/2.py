results = dict()
import paddle
import time
arg_1 = -2
arg_2 = 2048
arg_3_0 = 17
arg_3_1 = -35
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = -48
start = time.time()
results["time_low"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
results["time_high"] = time.time() - start

print(results)
