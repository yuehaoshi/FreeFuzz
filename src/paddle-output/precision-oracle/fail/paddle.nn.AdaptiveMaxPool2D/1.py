results = dict()
import paddle
import time
arg_1 = 3
arg_2 = False
arg_class = paddle.nn.AdaptiveMaxPool2D(output_size=arg_1,return_mask=arg_2,)
arg_3 = None
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
