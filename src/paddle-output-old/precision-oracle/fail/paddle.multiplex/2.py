results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-4096,4096,[1], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_tensor = paddle.randint(-1,64,[2, 1], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
start = time.time()
results["time_low"] = paddle.multiplex(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,]
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.multiplex(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
