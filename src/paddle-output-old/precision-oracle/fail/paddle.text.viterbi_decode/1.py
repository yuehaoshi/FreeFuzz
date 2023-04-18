results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-64,2,[2, 4, 3], dtype=paddle.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16,16,[2], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4 = False
start = time.time()
results["time_low"] = paddle.text.viterbi_decode(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.int16)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.text.viterbi_decode(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
