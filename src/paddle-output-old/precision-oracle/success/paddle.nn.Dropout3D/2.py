results = dict()
import paddle
import time
arg_1 = 0.5
start = time.time()
results["time_low"] = paddle.nn.Dropout3D(p=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.Dropout3D(p=arg_1,)
results["time_high"] = time.time() - start

print(results)
