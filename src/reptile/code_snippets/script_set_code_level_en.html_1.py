import paddle

paddle.jit.set_code_level(2)
# It will print the transformed code at level 2, which means to print the code after second transformer,
# as the date of August 28, 2020, it is CastTransformer.

os.environ['TRANSLATOR_CODE_LEVEL'] = '3'
# The code level is now 3, but it has no effect because it has a lower priority than `set_code_level`