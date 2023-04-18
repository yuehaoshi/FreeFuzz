results = dict()
import paddle
import time
arg_1 = 2
arg_2 = 2
arg_class = paddle.nn.MaxPool2D(arg_1,arg_2,)
arg_3_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
start = time.time()
results["time_low"] = arg_class(*arg_3)
results["time_low"] = time.time() - start
arg_3 = arg_3_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = arg_class(*arg_3)
results["time_high"] = time.time() - start

print(results)
