results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([4, 2, 1, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16384,32,[24, 2, 11], dtype=paddle.bfloat16)
arg_2 = arg_2_tensor.clone()
arg_3 = True
start = time.time()
results["time_low"] = paddle.matmul(x=arg_1,y=arg_2,transpose_y=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_2 = arg_2_tensor.clone().astype(paddle.bfloat16)
start = time.time()
results["time_high"] = paddle.matmul(x=arg_1,y=arg_2,transpose_y=arg_3,)
results["time_high"] = time.time() - start

print(results)
