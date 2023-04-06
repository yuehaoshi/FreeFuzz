import paddle.fluid as fluid

startup_prog = fluid.Program()
main_prog = fluid.Program()
with fluid.program_guard(startup_prog, main_prog):
    original_variable = fluid.data(name = "new_variable", shape=[2,2], dtype='float32')
    new_variable = original_variable.astype('int64')
    print("new var's dtype is: {}".format(new_variable.dtype))