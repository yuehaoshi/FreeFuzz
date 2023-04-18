results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
start = time.time()
results["time_low"] = paddle.scale(arg_1,scale=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.scale(arg_1,scale=arg_2,)
results["time_high"] = time.time() - start

print(results)
