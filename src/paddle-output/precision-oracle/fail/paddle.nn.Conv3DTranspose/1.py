results = dict()
import paddle
import time
arg_1 = -1024
arg_2 = 6
arg_3_0 = 3
arg_3_1 = 3
arg_3_2 = 3
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
arg_class = paddle.nn.Conv3DTranspose(arg_1,arg_2,arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
