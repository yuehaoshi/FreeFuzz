results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-256,16,[3, 4], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,2,[3], dtype=paddle.int8)
arg_2 = arg_2_tensor.clone()
arg_3 = 0
start = time.time()
results["time_low"] = paddle.index_select(x=arg_1,index=arg_2,axis=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.int32)
start = time.time()
results["time_high"] = paddle.index_select(x=arg_1,index=arg_2,axis=arg_3,)
results["time_high"] = time.time() - start

print(results)
