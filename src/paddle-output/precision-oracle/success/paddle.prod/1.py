results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = "int64"
start = time.time()
results["time_low"] = paddle.prod(arg_1,arg_2,dtype=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
start = time.time()
results["time_high"] = paddle.prod(arg_1,arg_2,dtype=arg_3,)
results["time_high"] = time.time() - start

print(results)
