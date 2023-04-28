results = dict()
import paddle
import time
arg_1 = "/var/folders/l7/ph7j62kx4sbd9fyvt_w0lqpc0000gn/T/tmprmdo5pyl/no_grad_infer_model"
start = time.time()
results["time_low"] = paddle.jit.load(arg_1,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.jit.load(arg_1,)
results["time_high"] = time.time() - start

print(results)
