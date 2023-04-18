results = dict()
import paddle
import time
arg_1 = "tanh"
arg_2_tensor = paddle.randint(-16,32768,[1, 200], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16384,4096,[1, 200], dtype=paddle.float16)
arg_3 = arg_3_tensor.clone()
arg_4 = "tanh_grad"
arg_5_tensor = paddle.randint(-2048,64,[1, 200], dtype=paddle.float16)
arg_5 = arg_5_tensor.clone()
start = time.time()
results["time_low"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,backward_func=arg_4,skip_vars_in_backward_input=arg_5,)
results["time_low"] = time.time() - start
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.float32)
arg_5 = arg_5_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,backward_func=arg_4,skip_vars_in_backward_input=arg_5,)
results["time_high"] = time.time() - start

print(results)
