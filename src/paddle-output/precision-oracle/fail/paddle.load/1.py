results = dict()
import paddle
import time
arg_1 = "/Users/huyiteng/.paddlenlp/models/__internal_testing__/tiny-random-ernie/model_state.pdparams"
arg_2 = True
start = time.time()
results["time_low"] = paddle.load(arg_1,return_numpy=arg_2,)
results["time_low"] = time.time() - start
start = time.time()
results["time_high"] = paddle.load(arg_1,return_numpy=arg_2,)
results["time_high"] = time.time() - start

print(results)
