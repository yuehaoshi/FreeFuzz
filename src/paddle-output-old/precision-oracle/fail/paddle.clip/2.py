results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,8192,[2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = 5.0
start = time.time()
results["time_low"] = paddle.clip(arg_1,min=arg_2,max=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.clip(arg_1,min=arg_2,max=arg_3,)
results["time_high"] = time.time() - start

print(results)
