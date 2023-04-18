results = dict()
import paddle
import time
arg_1_tensor = paddle.randint(-16384,32,[8, 48000], dtype=paddle.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 511
arg_3 = True
start = time.time()
results["time_low"] = paddle.signal.stft(arg_1,n_fft=arg_2,onesided=arg_3,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float64)
start = time.time()
results["time_high"] = paddle.signal.stft(arg_1,n_fft=arg_2,onesided=arg_3,)
results["time_high"] = time.time() - start

print(results)
