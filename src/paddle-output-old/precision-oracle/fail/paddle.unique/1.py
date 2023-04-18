results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,32,[6], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2 = True
arg_3 = True
arg_4 = True
start = time.time()
results["time_low"] = paddle.unique(arg_1,return_index=arg_2,return_inverse=arg_3,return_counts=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.unique(arg_1,return_index=arg_2,return_inverse=arg_3,return_counts=arg_4,)
results["time_high"] = time.time() - start

print(results)
