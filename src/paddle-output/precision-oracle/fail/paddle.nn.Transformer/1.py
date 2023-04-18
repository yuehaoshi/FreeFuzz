results = dict()
import paddle
import time
arg_1 = 128
arg_2 = 2
arg_3 = 29
arg_4 = 64
arg_5 = 512
arg_class = paddle.nn.Transformer(arg_1,arg_2,arg_3,arg_4,arg_5,)
arg_6 = "max"
start = time.time()
results["time_low"] = arg_class(*arg_6)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_6)
results["time_high"] = time.time() - start

print(results)
