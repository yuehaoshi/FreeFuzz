results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 4, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -16.0
start = time.time()
results["time_low"] = paddle.unbind(arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.unbind(arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)
