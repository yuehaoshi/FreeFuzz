results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-8,1,[2, 3], dtype=paddle.int8)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-64,1,[2, 3], dtype=paddle.int8)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2_tensor = paddle.rand([2, 3, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.bmm(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.int32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.int32)
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.bmm(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
