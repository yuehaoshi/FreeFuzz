results = dict()
import paddle
import time
arg_1 = 4
arg_2 = 6
arg_3_0 = 16
arg_3_1 = -35
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.nn.Conv2DTranspose(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.nn.Conv2DTranspose(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
