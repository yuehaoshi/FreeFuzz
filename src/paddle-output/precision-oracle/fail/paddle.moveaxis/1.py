results = dict()
import paddle
import time
real = paddle.rand([2, 0], paddle.float32)
imag = paddle.rand([2, 0], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = -9
start = time.time()
results["time_low"] = paddle.moveaxis(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.complex64)
start = time.time()
results["time_high"] = paddle.moveaxis(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

print(results)
