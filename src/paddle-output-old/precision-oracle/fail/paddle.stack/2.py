results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-2048,32768,[1, 2], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-32768,8,[1, 2], dtype=paddle.float16)
arg_1_1 = arg_1_1_tensor.clone()
arg_1_2_tensor = paddle.randint(-32,128,[1, 2], dtype=paddle.float16)
arg_1_2 = arg_1_2_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = 0
start = time.time()
results["time_low"] = paddle.stack(arg_1,axis=arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1_2 = arg_1_2_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
start = time.time()
results["time_high"] = paddle.stack(arg_1,axis=arg_2,)
results["time_high"] = time.time() - start

print(results)