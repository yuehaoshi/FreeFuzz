results = dict()
import paddle
import time
arg_1 = "batchmean"
start = time.time()
results["time_low"] = paddle.nn.KLDivLoss(reduction=arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.KLDivLoss(reduction=arg_1,)
results["time_high"] = time.time() - start

print(results)
