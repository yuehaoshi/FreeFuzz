import paddle

prog_trans = paddle.jit.ProgramTranslator()
prog_cache = prog_trans.get_program_cache()