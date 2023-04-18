results = dict()
import paddle
import time
arg_1_0_tensor = paddle.randint(-4096,256,[1], dtype=paddle.float16)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-128,64,[1], dtype=paddle.float16)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0_tensor = paddle.randint(-16,2,[1], dtype=paddle.float16)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3 = None
start = time.time()
results["time_low"] = paddle.grad(outputs=arg_1,inputs=arg_2,grad_outputs=arg_3,)
results["time_low"] = time.time() - start
arg_1_0 = arg_1_0_tensor.clone().type(paddle.float32)
arg_1_1 = arg_1_1_tensor.clone().type(paddle.float32)
arg_1 = [arg_1_0,arg_1_1,]
arg_2_0 = arg_2_0_tensor.clone().type(paddle.float32)
arg_2 = [arg_2_0,]
start = time.time()
results["time_high"] = paddle.grad(outputs=arg_1,inputs=arg_2,grad_outputs=arg_3,)
results["time_high"] = time.time() - start

print(results)
