results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -inf
start = time.time()
results["time_low"] = paddle.linalg.cond(arg_1,p=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.linalg.cond(arg_1,p=arg_2,)
results["time_high"] = time.time() - start

print(results)
