results = dict()
import paddle
import time
arg_1 = 55.0
arg_2 = -15
arg_3 = 516
arg_class = paddle.nn.TransformerDecoderLayer(arg_1,arg_2,arg_3,)
arg_4 = None
start = time.time()
results["time_low"] = arg_class(*arg_4)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_4)
results["time_high"] = time.time() - start

print(results)
