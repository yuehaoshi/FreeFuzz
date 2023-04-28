import paddle
arg_1 = "/Users/ashi_mac/VSC/CS527/Paddle/test/wave_test.wav"
arg_2_tensor = paddle.rand([1, 8000], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 16000
arg_4 = True
res = paddle.audio.save(arg_1,arg_2,arg_3,channels_first=arg_4,)
