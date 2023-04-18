results = dict()
import paddle
import time
arg_1_0_tensor = paddle.rand([2, 3, 4], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-8,32,[2, 1], dtype=paddle.int8)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_class = paddle.io.TensorDataset(arg_1,)
arg_2 = "max"
start = time.time()
results["time_low"] = arg_class(*arg_2)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.int32)
arg_1 = [arg_1_0,arg_1_1,]
start = time.time()
results["time_high"] = arg_class(*arg_2)
results["time_high"] = time.time() - start

print(results)
