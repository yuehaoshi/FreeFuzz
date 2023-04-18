import paddle

# Two methods get same object because ProgramTranslator is a singleton
paddle.jit.ProgramTranslator()
paddle.jit.ProgramTranslator.get_instance()