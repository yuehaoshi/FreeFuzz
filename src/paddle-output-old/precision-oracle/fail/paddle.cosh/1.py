results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.cosh(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.cosh(arg_1,)
results["time_high"] = time.time() - start

print(results)
