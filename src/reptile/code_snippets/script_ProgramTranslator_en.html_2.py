import paddle


@paddle.jit.to_static
def func(x):
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


prog_trans = paddle.jit.ProgramTranslator()
prog_trans.enable(False)

x = paddle.ones([1, 2])
# ProgramTranslator is disabled so the func is run in dygraph
print(func(x))  # [[0. 0.]]