import paddle


def func(x):
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


prog_trans = paddle.jit.ProgramTranslator()

x = paddle.ones([1, 2])
x_v = prog_trans.get_output(func, x)
print(x_v)  # [[0. 0.]]