results = dict()
import paddle
import time
arg_1 = False
arg_2 = 1024
arg_3 = 2.0
arg_4 = 16.5
arg_5 = 1000
arg_6 = 2
arg_7 = True
arg_class = paddle.amp.GradScaler(enable=arg_1,init_loss_scaling=arg_2,incr_ratio=arg_3,decr_ratio=arg_4,incr_every_n_steps=arg_5,decr_every_n_nan_or_inf=arg_6,use_dynamic_loss_scaling=arg_7,)
arg_8 = None
start = time.time()
results["time_low"] = arg_class(*arg_8)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = arg_class(*arg_8)
results["time_high"] = time.time() - start

print(results)
