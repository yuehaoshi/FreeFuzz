results = dict()
import paddle
import time
arg_1_0 = -1024
arg_1_1 = 13
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.nn.UpsamplingBilinear2D(size=arg_1,)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.nn.UpsamplingBilinear2D(size=arg_1,)
results["time_high"] = time.time() - start

print(results)
