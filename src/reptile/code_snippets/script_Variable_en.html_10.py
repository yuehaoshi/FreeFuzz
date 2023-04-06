import paddle.fluid as fluid
cur_program = fluid.Program()
cur_block = cur_program.current_block()
new_variable = cur_block.create_var(name="X",
                                    shape=[-1, 23, 48],
                                    dtype='float32')
print("persistable of current Var is: {}".format(new_variable.persistable))