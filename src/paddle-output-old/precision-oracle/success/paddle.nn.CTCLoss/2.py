results = dict()
import paddle
import time
arg_1 = -52
arg_2 = "mean"
start = time.time()
results["time_low"] = paddle.nn.CTCLoss(blank=arg_1,reduction=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.nn.CTCLoss(blank=arg_1,reduction=arg_2,)
results["time_high"] = time.time() - start

print(results)
