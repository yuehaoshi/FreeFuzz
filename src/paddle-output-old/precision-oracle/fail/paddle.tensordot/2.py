results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-8,8,[2, 3, 4, 5, 6], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,128,[3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_0_0 = 0
arg_3_0_1 = 1
arg_3_0_2 = 3
arg_3_0_3 = 4
arg_3_0 = [arg_3_0_0,arg_3_0_1,arg_3_0_2,arg_3_0_3,]
arg_3_1_0 = 1
arg_3_1_1 = 0
arg_3_1 = [arg_3_1_0,arg_3_1_1,]
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_low"] = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3_0 = [arg_3_0_0,arg_3_0_1,arg_3_0_2,arg_3_0_3,]
arg_3_1 = [arg_3_1_0,arg_3_1_1,]
arg_3 = [arg_3_0,arg_3_1,]
start = time.time()
results["time_high"] = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
results["time_high"] = time.time() - start

print(results)