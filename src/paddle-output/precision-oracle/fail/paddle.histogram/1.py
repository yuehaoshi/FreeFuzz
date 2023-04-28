results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([1, 1, 28, 28], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([1, 400], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 4
arg_3 = 0
arg_4 = 3
start = time.time()
results["time_low"] = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().astype(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().astype(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
results["time_high"] = time.time() - start

print(results)
