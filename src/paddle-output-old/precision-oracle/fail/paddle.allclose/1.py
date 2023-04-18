results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3 = 1e-05
arg_4 = 1e-08
arg_5 = False
arg_6 = None
start = time.time()
results["time_low"] = paddle.allclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.allclose(arg_1,arg_2,rtol=arg_3,atol=arg_4,equal_nan=arg_5,name=arg_6,)
results["time_high"] = time.time() - start

print(results)
