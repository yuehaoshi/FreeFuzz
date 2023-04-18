results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-2,8192,[3, 10, 5, 10], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = 1
arg_4 = -1
start = time.time()
results["time_low"] = paddle.trace(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.trace(arg_1,offset=arg_2,axis1=arg_3,axis2=arg_4,)
results["time_high"] = time.time() - start

print(results)
