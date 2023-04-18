results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.5
arg_3 = -1
start = time.time()
results["time_low"] = paddle.norm(arg_1,p=arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.norm(arg_1,p=arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)
