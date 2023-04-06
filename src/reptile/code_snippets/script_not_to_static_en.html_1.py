import paddle

@paddle.jit.not_to_static
def func_not_to_static(x):
    res = x - 1
    return res

@paddle.jit.to_static
def func(x):
    if paddle.mean(x) < 0:
        out = func_not_to_static(x)
    else:
        out = x + 1
    return out

x = paddle.ones([1, 2], dtype='float32')
out = func(x)
print(out) # [[2. 2.]]