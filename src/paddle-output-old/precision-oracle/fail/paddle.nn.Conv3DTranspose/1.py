results = dict()
import paddle
import time
arg_1 = 4
arg_2 = 6
arg_3_0 = -56
arg_3_1 = 77
arg_3_2 = -29
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
start = time.time()
results["time_low"] = paddle.nn.Conv3DTranspose(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
start = time.time()
results["time_high"] = paddle.nn.Conv3DTranspose(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)