results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "double_hook_fn"
start = time.time()
results["time_low"] = paddle.Tensor.register_hook(arg_1,arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.Tensor.register_hook(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
