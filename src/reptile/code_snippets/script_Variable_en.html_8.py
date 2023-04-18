import paddle.fluid as fluid
import paddle

paddle.enable_static()
cur_program = fluid.Program()
cur_block = cur_program.current_block()
new_variable = cur_block.create_var(name="X",
                                    shape=[-1, 23, 48],
                                    dtype='float32')
print(new_variable.to_string(True))
print("=============with detail===============")
print(new_variable.to_string(True, True))