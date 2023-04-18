import paddle


def func(x):
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v


prog_trans = paddle.jit.ProgramTranslator()
x = paddle.ones([1, 2])
main_prog, start_prog, inputs, outputs = prog_trans.get_program(func, x)
print([i.name for i in inputs])
# [u'generated_tensor_0'] the feed input Tensor name representing x
print([o.name for o in outputs])
# [u'_generated_var_4'] the fetch output Tensor name representing x_v