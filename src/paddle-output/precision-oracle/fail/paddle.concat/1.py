results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-2,32,[2, 3], dtype=paddle.int8)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8,128,[2, 3], dtype=paddle.int8)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.randint(-64,32,[2, 2], dtype=paddle.int8)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = 1
start = time.time()
results["time_low"] = paddle.concat(x=arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.int64)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.int64)
arg_1_2 = arg_1_2_tensor.clone().type(paddle.int64)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_high"] = paddle.concat(x=arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)
