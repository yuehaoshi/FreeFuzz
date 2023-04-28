results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 9, 6, 5, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
start = time.time()
results["time_low"] = paddle.Tensor.uniform_(arg_1,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float64)
start = time.time()
results["time_high"] = paddle.Tensor.uniform_(arg_1,)
results["time_high"] = time.time() - start

print(results)
