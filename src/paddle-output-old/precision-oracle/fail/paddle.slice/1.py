results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([1, 2, 3, 4], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([1, 2, 1, 4], dtype=paddle.float16)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.rand([1, 1, 3, 1], dtype=paddle.float16)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2_0 = 0
arg_2 = [arg_2_0,]
arg_3_0 = 1
arg_3 = [arg_3_0,]
arg_4_0 = 2
arg_4 = [arg_4_0,]
start = time.time()
results["time_low"] = paddle.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1_2 = arg_1_2_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = [arg_2_0,]
arg_3 = [arg_3_0,]
arg_4 = [arg_4_0,]
start = time.time()
results["time_high"] = paddle.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
results["time_high"] = time.time() - start

print(results)
