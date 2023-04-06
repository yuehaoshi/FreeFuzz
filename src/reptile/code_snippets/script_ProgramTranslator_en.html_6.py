import paddle


def func(x):
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


prog_trans = paddle.jit.ProgramTranslator()

code = prog_trans.get_code(func)
print(type(code)) # <class 'str'>