results = dict()
import paddle
import time
arg_1 = -1.0
start = time.time()
results["time_low"] = paddle.nn.ClipGradByNorm(clip_norm=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.ClipGradByNorm(clip_norm=arg_1,)
results["time_high"] = time.time() - start

print(results)