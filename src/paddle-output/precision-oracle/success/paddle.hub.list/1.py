results = dict()
import paddle
import time
arg_1 = "lyuwenyu/paddlehub_demo:main"
arg_2 = "github"
arg_3 = 76.0
start = time.time()
results["time_low"] = paddle.hub.list(arg_1,source=arg_2,force_reload=arg_3,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.hub.list(arg_1,source=arg_2,force_reload=arg_3,)
results["time_high"] = time.time() - start

print(results)
