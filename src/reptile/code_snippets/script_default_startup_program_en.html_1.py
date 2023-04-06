import paddle

paddle.enable_static()
x = paddle.static.data(name="x", shape=[-1, 784], dtype='float32')
out = paddle.static.nn.fc(name="fc", x=x, size=10, activation="relu")
print("main program is: {}".format(paddle.static.default_main_program()))
print("start up program is: {}".format(paddle.static.default_startup_program()))