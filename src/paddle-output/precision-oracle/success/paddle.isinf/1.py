results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([7], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.isinf(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.isinf(arg_1,)
results["time_high"] = time.time() - start

print(results)
