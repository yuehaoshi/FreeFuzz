results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-64,2048,[2, 3, 4], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-64,4,[2, 1], dtype=paddle.int8)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_low"] = paddle.io.TensorDataset(arg_1,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.int32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = paddle.io.TensorDataset(arg_1,)
results["time_high"] = time.time() - start

print(results)
