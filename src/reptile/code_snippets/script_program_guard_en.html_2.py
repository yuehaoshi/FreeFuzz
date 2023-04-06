import paddle

paddle.enable_static()
main_program = paddle.static.Program()
# does not care about startup program. Just pass a temporary value.
with paddle.static.program_guard(main_program, paddle.static.Program()):
    data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')