results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-4,2048,[-1, 28, 28], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_0_tensor = paddle.randint(-128,2048,[-1, 10], dtype=paddle.float16)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
start = time.time()
results["time_low"] = paddle.static.serialize_program(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,]
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.static.serialize_program(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
