results = dict()
import paddle
import time
arg_1_0 = 2
arg_1_1 = 8
arg_1_2 = 32
arg_1_3 = 32
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = 1
arg_3 = 30
start = time.time()
results["time_low"] = paddle.nn.SpectralNorm(arg_1,dim=arg_2,power_iters=arg_3,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
start = time.time()
results["time_high"] = paddle.nn.SpectralNorm(arg_1,dim=arg_2,power_iters=arg_3,)
results["time_high"] = time.time() - start

print(results)
