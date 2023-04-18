results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-1,16,[3, 2], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 9, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 2
arg_3_1 = 3
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.scatter_nd(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int64)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.scatter_nd(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
