results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = False
start = time.time()
results["time_low"] = paddle.text.ViterbiDecoder(arg_1,include_bos_eos_tag=arg_2,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.text.ViterbiDecoder(arg_1,include_bos_eos_tag=arg_2,)
results["time_high"] = time.time() - start

print(results)
