results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-256,2048,[2, 3, 5], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 44.0
start = time.time()
results["time_low"] = paddle.full_like(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.full_like(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
