results = dict()
import paddle
import time
arg_1 = False
start = time.time()
results["time_low"] = paddle.ParamAttr(need_clip=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.ParamAttr(need_clip=arg_1,)
results["time_high"] = time.time() - start

print(results)
