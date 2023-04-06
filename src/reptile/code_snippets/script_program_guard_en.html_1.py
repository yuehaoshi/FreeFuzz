import paddle

paddle.enable_static()
main_program = paddle.static.Program()
startup_program = paddle.static.Program()
with paddle.static.program_guard(main_program, startup_program):
    data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
    hidden = paddle.static.nn.fc(x=data, size=10, activation='relu')