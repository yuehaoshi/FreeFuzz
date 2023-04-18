results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,2,[3, 4], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = False
arg_3 = True
start = time.time()
results["time_low"] = paddle.median(arg_1,axis=arg_2,keepdim=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.median(arg_1,axis=arg_2,keepdim=arg_3,)
results["time_high"] = time.time() - start

print(results)
