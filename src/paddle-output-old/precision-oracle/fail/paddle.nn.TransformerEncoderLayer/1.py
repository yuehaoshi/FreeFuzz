results = dict()
import paddle
import time
arg_1 = 151
arg_2 = 2
arg_3 = 512
start = time.time()
results["time_low"] = paddle.nn.TransformerEncoderLayer(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.TransformerEncoderLayer(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)