results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 38
start = time.time()
results["time_low"] = paddle.logsumexp(arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.logsumexp(arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)
