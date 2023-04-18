results = dict()
import paddle
import time
arg_1_0 = -1024
arg_1_1 = 1024
arg_1_2 = -1
arg_1_3 = -17
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = -12
arg_3 = 30
arg_class = paddle.nn.SpectralNorm(arg_1,dim=arg_2,power_iters=arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
