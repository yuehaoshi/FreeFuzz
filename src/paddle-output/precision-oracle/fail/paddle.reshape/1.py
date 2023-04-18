results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 4, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 5
arg_2_2 = 9
arg_2_3 = 10
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_low"] = paddle.reshape(arg_1,shape=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
start = time.time()
results["time_high"] = paddle.reshape(arg_1,shape=arg_2,)
results["time_high"] = time.time() - start

print(results)
