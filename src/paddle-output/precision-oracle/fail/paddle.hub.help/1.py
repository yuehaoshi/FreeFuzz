results = dict()
import paddle
import time
arg_1 = "lyuwenyu/paddlehub_demo:main"
arg_2 = "zeros"
arg_3 = -54
start = time.time()
results["time_low"] = paddle.hub.help(arg_1,model=arg_2,source=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.hub.help(arg_1,model=arg_2,source=arg_3,)
results["time_high"] = time.time() - start

print(results)
