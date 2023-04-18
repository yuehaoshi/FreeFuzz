results = dict()
import paddle
import time
arg_1 = True
arg_2 = 1024
arg_3 = 2.0
arg_4 = -44.5
arg_5 = 1024
arg_6 = 2
arg_7 = True
start = time.time()
results["time_low"] = paddle.amp.GradScaler(enable=arg_1,init_loss_scaling=arg_2,incr_ratio=arg_3,decr_ratio=arg_4,incr_every_n_steps=arg_5,decr_every_n_nan_or_inf=arg_6,use_dynamic_loss_scaling=arg_7,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.amp.GradScaler(enable=arg_1,init_loss_scaling=arg_2,incr_ratio=arg_3,decr_ratio=arg_4,incr_every_n_steps=arg_5,decr_every_n_nan_or_inf=arg_6,use_dynamic_loss_scaling=arg_7,)
results["time_high"] = time.time() - start

print(results)
