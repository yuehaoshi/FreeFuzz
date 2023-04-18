import paddle

paddle.enable_static()

x = paddle.full(shape=[2, 3], fill_value=3, dtype='int64')
out = paddle.static.Print(x, message="The content of input layer:")

main_program = paddle.static.default_main_program()
exe = paddle.static.Executor(place=paddle.CPUPlace())
res = exe.run(main_program, fetch_list=[out])
# Variable: fill_constant_1.tmp_0
#   - message: The content of input layer:
#   - lod: {}
#   - place: CPUPlace
#   - shape: [2, 3]
#   - layout: NCHW
#   - dtype: long
#   - data: [3 3 3 3 3 3]