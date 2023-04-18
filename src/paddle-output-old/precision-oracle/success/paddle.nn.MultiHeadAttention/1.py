results = dict()
import paddle
import time
arg_1 = 128
arg_2 = 2
start = time.time()
results["time_low"] = paddle.nn.MultiHeadAttention(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.MultiHeadAttention(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
