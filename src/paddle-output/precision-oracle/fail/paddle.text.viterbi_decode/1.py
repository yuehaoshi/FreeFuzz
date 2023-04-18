results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([2, 4, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-8,64,[2], dtype=paddle.int8)
arg_3 = arg_3_tensor.clone()
arg_4 = "replicate"
start = time.time()
results["time_low"] = paddle.text.viterbi_decode(arg_1,arg_2,arg_3,arg_4,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_2 = arg_2_tensor.clone().type(paddle.float32)
arg_3 = arg_3_tensor.clone().type(paddle.int64)
start = time.time()
results["time_high"] = paddle.text.viterbi_decode(arg_1,arg_2,arg_3,arg_4,)
results["time_high"] = time.time() - start

print(results)
