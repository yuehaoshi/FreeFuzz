results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-32768,2048,[2, 3, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 0
arg_2_2 = 2
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_low"] = paddle.transpose(arg_1,perm=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
start = time.time()
results["time_high"] = paddle.transpose(arg_1,perm=arg_2,)
results["time_high"] = time.time() - start

print(results)
