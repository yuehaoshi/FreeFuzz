results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,16384,[5], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = -18
arg_3 = None
start = time.time()
results["time_low"] = paddle.roll(arg_1,arg_2,name=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.roll(arg_1,arg_2,name=arg_3,)
results["time_high"] = time.time() - start

print(results)
