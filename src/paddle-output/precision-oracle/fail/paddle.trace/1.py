results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 27
arg_3 = 1
arg_4 = 2
start = time.time()
results["time_low"] = paddle.trace(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.trace(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
results["time_high"] = time.time() - start

print(results)
