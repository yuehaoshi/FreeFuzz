results = dict()
import paddle
import time
arg_1_0 = 3
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_class = paddle.nn.UpsamplingBilinear2D(size=arg_1,)
arg_2 = None
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
