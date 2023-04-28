results = dict()
import paddle
import time
arg_1 = "paddlenlp.transformers.ernie.modelingUTC"
arg_2 = "/var/folders/_g/fgwmk63n7y57_5d31qhhk82h0000gn/T/tmpq0h_phmw/1682372204/f14714bc/exported_model/model"
start = time.time()
results["time_low"] = paddle.jit.save(arg_1,arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.jit.save(arg_1,arg_2,)
results["time_high"] = time.time() - start

print(results)
