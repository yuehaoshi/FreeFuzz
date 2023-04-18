results = dict()
import paddle
import time
arg_1 = 128
arg_2 = 2
arg_3 = 516
start = time.time()
results["time_low"] = paddle.nn.TransformerDecoderLayer(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.TransformerDecoderLayer(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
