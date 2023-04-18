results = dict()
import paddle
import time
arg_1 = 128
arg_2 = 2
arg_3 = 29
arg_4 = 28.0
arg_5 = 1026
start = time.time()
results["time_low"] = paddle.nn.Transformer(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.Transformer(arg_1,arg_2,arg_3,arg_4,arg_5,)
results["time_high"] = time.time() - start

print(results)
