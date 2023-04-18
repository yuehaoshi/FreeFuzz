results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-8192,512,[-1, 4, 8, 8], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_tensor = paddle.randint(-1024,2048,[-1, 2, 8, 8], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.static.gradients(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,]
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.gradients(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
